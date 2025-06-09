from dataclasses import dataclass, field
import numpy as np
import random
import math
from copy import deepcopy
from ofcp_player import OpenFaceChinesePoker
import pickle, pathlib
from mcts import MCTSAgent
from tqdm import tqdm

def save_regret_table(path: str = "cfr_table.pkl"):
    """Store REGRET_TABLE to disk."""
    with pathlib.Path(path).open("wb") as fh:
        pickle.dump(REGRET_TABLE, fh)

def load_regret_table(path: str = "cfr_table.pkl"):
    """Load REGRET_TABLE from disk (over-writes current content)."""
    global REGRET_TABLE
    with pathlib.Path(path).open("rb") as fh:
        REGRET_TABLE = pickle.load(fh)

ACTIONS = ['top', 'middle', 'bottom']

@dataclass
class RegretNode:
    regret_sum:   np.ndarray = field(default_factory=lambda: np.zeros(len(ACTIONS)))
    strategy_sum: np.ndarray = field(default_factory=lambda: np.zeros(len(ACTIONS)))

    def strategy(self, realization_weight: float) -> np.ndarray:
        # CFR+ uses positive regrets; vanilla CFR uses raw regrets (can be negative)
        positive_regrets = np.maximum(self.regret_sum, 0.0)
        if positive_regrets.sum() > 0:
            strat = positive_regrets / positive_regrets.sum()
        else:                       # fallback to uniform
            strat = np.ones(len(ACTIONS)) / len(ACTIONS)
        self.strategy_sum += realization_weight * strat
        return strat

    def average_strategy(self) -> np.ndarray:
        if self.strategy_sum.sum() == 0:
            return np.ones(len(ACTIONS)) / len(ACTIONS)
        return self.strategy_sum / self.strategy_sum.sum()
    
REGRET_TABLE: dict[str, RegretNode] = {}


def infoset_key(state, card_to_place) -> str:
    # Serialize only *public* information from the current player’s viewpoint
    # (own piles are public, remaining deck size is public, the actual deck order is hidden)
    key_parts = [
        ''.join(str(card) for card in state.player_hand.top),
        ''.join(str(card) for card in state.player_hand.middle),
        ''.join(str(card) for card in state.player_hand.bottom),
        str(len(state.deck.cards)),              # abstraction: just its length
        str(card_to_place)                 # the card we must place now
    ]
    return '|'.join(key_parts)


def cfr(state, card_to_place, p_player: float, p_opponent: float, iteration: int) -> float:
    """
    Returns utility for current player under optimal play,
    while updating regrets / strategies on the way back up.
    p_player   : reach prob of *this* player’s infoset so far
    p_opponent : reach prob of opponent’s nodes so far
    """

    if state.game_over():                 # terminal
        player_score, bot_score = state.calculate_scores()
        return player_score - bot_score   # utility from current player's POV

    key = infoset_key(state, card_to_place)
    node = REGRET_TABLE.setdefault(key, RegretNode())

    # Current mixed strategy
    sigma = node.strategy(p_player)

    # Utilities per action and node utility
    util = np.zeros(len(ACTIONS))
    node_util = 0.0

    for a_idx, pos in enumerate(ACTIONS):
        if len(getattr(state.player_hand, pos)) >= (3 if pos == 'top' else 5):
            continue  # illegal, keep util=0

        next_state = deepcopy(state)
        bot_card   = next_state.deck.draw(1)            # still random for opponent
        next_state.play_round(
            {'cards': [card_to_place], 'positions': [(pos, card_to_place)]},
            CFRPolicyBot().choose(next_state, bot_card[0]))

        next_player_card, _unused_bot = next_state.deal_next_cards()

        util[a_idx] = -cfr(next_state, next_player_card[0],  # switch perspective
                           p_opponent, p_player * sigma[a_idx],
                           iteration)
        node_util += sigma[a_idx] * util[a_idx]

    # --- regret update (CFR+) ---
    regret_delta = util - node_util
    node.regret_sum += (p_opponent * regret_delta)

    return node_util


def train_cfr(num_iters=10_000):
    for it in tqdm(range(1, num_iters + 1)):
        game = OpenFaceChinesePoker()
        player_card, _ = game.deal_next_cards()
        _ = cfr(game, player_card[0], p_player=1.0, p_opponent=1.0, iteration=it)

        if it % 500 == 0:
            exploitability = evaluate_current_strategy(200)
            print(f"[iter {it}] infosets={len(REGRET_TABLE)}, exploit={exploitability:.2f}")


class CFRPolicyBot:
    def choose(self, state, card_to_place):
        key = infoset_key(state, card_to_place)
        if key in REGRET_TABLE:
            probs = REGRET_TABLE[key].average_strategy()
        else:                       # unseen infoset ⇒ uniform
            probs = np.ones(len(ACTIONS)) / len(ACTIONS)
        pos = random.choices(ACTIONS, weights=probs, k=1)[0]
        return {'cards': [card_to_place], 'positions': [(pos, card_to_place)]}


def evaluate_current_strategy(num_games: int = 100,
                              mcts_sims: int = 2000,
                              use_rave: bool = True) -> float:
    """
    Approximates exploitability:
    * Player A = fixed CFR average strategy.
    * Player B = strong MCTS player that *knows* A is fixed.
    Return: average score from Bs point of view
            (≈ how much a greedy searcher can exploit A).
    """
    cfr_bot   = CFRPolicyBot()
    mcts_bot  = MCTSAgent(num_simulations=mcts_sims, cfr=True)

    total_score_for_MCTS = 0.0

    for _ in range(num_games):
        game = OpenFaceChinesePoker()

        while not game.game_over():
            player_card, bot_card = game.deal_next_cards()

            # A (CFR) moves first
            a_move = cfr_bot.choose(game, player_card[0])

            # B (MCTS) searches from *current* position
            b_move_pos = mcts_bot.search(game, bot_card[0], rave=use_rave)
            b_move = {'cards': [bot_card[0]],
                      'positions': [(b_move_pos, bot_card[0])]} if b_move_pos else {'cards': [], 'positions': []}

            if a_move['positions'] and b_move['positions']:
                game.play_round(a_move, b_move)
            else:
                break

        a_pts, b_pts = game.calculate_scores()
        total_score_for_MCTS += (b_pts - a_pts)

    # positive ⇒ A is exploitable; negative ⇒ MCTS couldn’t beat it
    return total_score_for_MCTS / num_games


if __name__ == "__main__":
    train_cfr(num_iters=8)
    save_regret_table("cfr_table.pkl")