import random
import pickle
from ofcp_player import OpenFaceChinesePoker, random_bot_agent
from collections import defaultdict
from tqdm import tqdm
import numpy as np

NUM_EPISODES = 500

with open("Q-Learning/q_table.pkl", "rb") as f:
    Q = defaultdict(float, pickle.load(f))

positions = ['top', 'middle', 'bottom']
slot_limits = {'top': 3, 'middle': 5, 'bottom': 5}

def get_discrete_state(player_hand, current_card):
    rank_to_index = {
        '2': 0, '3': 1, '4': 2, '5': 3, '6': 4,
        '7': 5, '8': 6, '9': 7, 'T': 8,
        'J': 9, 'Q': 10, 'K': 11, 'A': 12
    }
    suit_to_index = {
        'C': 0, 'D': 1, 'H': 2, 'S': 3
    }

    def encode_card(card):
        vec = np.zeros(52)
        if card:
            rank_idx = rank_to_index.get(card.rank)
            suit_idx = suit_to_index.get(card.suit)
            if rank_idx is not None and suit_idx is not None:
                vec[suit_idx * 13 + rank_idx] = 1
        return vec

    state_vec = []
    for zone in ['top', 'middle', 'bottom']:
        cards = getattr(player_hand, zone)
        for card in cards:
            state_vec.extend(encode_card(card))
        max_len = 3 if zone == 'top' else 5
        for _ in range(max_len - len(cards)):
            state_vec.extend([0] * 52)

    state_vec.extend(encode_card(current_card))
    return tuple(state_vec)  # tuple for hashable key

def get_valid_actions(player_hand):
    return [i for i, pos in enumerate(positions)
            if len(getattr(player_hand, pos)) < slot_limits[pos]]

def select_action(state, valid_actions):
    if not valid_actions:
        return None  # No valid action available
    q_vals = [Q[(state, a)] for a in valid_actions]
    max_q = max(q_vals)
    best_actions = [a for a, q in zip(valid_actions, q_vals) if q == max_q]
    return random.choice(best_actions)


def evaluation():
    total_score = 0
    wins = 0

    total_p_points = 0
    total_b_points = 0
    total_p_wins = 0
    total_b_wins = 0
    num_games = NUM_EPISODES
    log_file = "Q-Learning/evaluation_results.txt"

    for _ in tqdm(range(NUM_EPISODES)):
        game = OpenFaceChinesePoker()
        player_initial, bot_initial = game.initial_deal()

        player_move = {
            'cards': player_initial,
            'positions': [(random.choice(positions), c) for c in player_initial]
        }
        
        bot_move = {
            'cards': bot_initial,
            'positions': [('bottom', c) for c in bot_initial]
        }
        game.play_round(player_move, bot_move)

        while not game.game_over() and len(game.deck.cards) >= 2:
            player_card, bot_card = game.deal_next_cards()
            state = get_discrete_state(game.player_hand, player_card[0])
            valid_actions = get_valid_actions(game.player_hand)

            # choosing action based off of training on the q-table
            action = select_action(state, valid_actions)

            if action is None:
                continue

            chosen_pos = positions[action]
            player_move = {
                'cards': [player_card[0]],
                'positions': [(chosen_pos, player_card[0])]
            }

            bot_play = random_bot_agent(game.bot_hand, [bot_card[0]], game.player_hand)
            game.play_round(player_move, bot_play)

        player_score, bot_score = game.calculate_scores()
        total_score += player_score
        total_p_points += player_score
        total_b_points += bot_score
        if player_score > bot_score:
            wins += 1
            total_p_wins += 1
        elif bot_score > player_score:
            total_b_wins += 1

    avg_score = total_score / NUM_EPISODES
    win_rate = wins / NUM_EPISODES

    print(f"\nEvaluation over {NUM_EPISODES} games:")
    print(f"Avg Player Score: {avg_score:.2f}")
    print(f"Win Rate: {win_rate * 100:.1f}%")

    # writing in log
    with open(log_file, "w") as f:
        f.write(f"\n--- QL {NUM_EPISODES} Evaluation Summary ---\n")
        f.write(f"Total QL Points: {total_p_points}\n")
        f.write(f"Total Bot Points: {total_b_points}\n")
        f.write(f"Average QL Points: {total_p_points / num_games:.2f}\n")
        f.write(f"Average Bot Points: {total_b_points / num_games:.2f}\n")
        f.write(f"QL Wins: {total_p_wins}/{num_games} ({100 * total_p_wins / num_games:.1f}%)\n")
        f.write(f"Bot Wins: {total_b_wins}/{num_games} ({100 * total_b_wins / num_games:.1f}%)\n")

    print("\nEvaluation results saved to:", log_file)

if __name__ == "__main__":
    evaluation()