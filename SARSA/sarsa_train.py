import random
import pickle
from collections import defaultdict
from ofcp_player import OpenFaceChinesePoker, random_bot_agent
import numpy as np

positions = ['top', 'middle', 'bottom']
max_slots = {'top': 3, 'middle': 5, 'bottom': 5}

Q = defaultdict(float)

ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.1

EPISODES = 1000

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
    return tuple(state_vec)

def get_valid_actions(player_hand):
    return [i for i, pos in enumerate(positions)
            if len(getattr(player_hand, pos)) < max_slots[pos]]

for ep in range(EPISODES):
    game = OpenFaceChinesePoker()
    player_initial, bot_initial = game.initial_deal()

    random.shuffle(player_initial)
    player_move = {
        'cards': player_initial,
        'positions': [(random.choice(positions), card) for card in player_initial]
    }
    bot_move = {'cards': bot_initial, 'positions': [('bottom', card) for card in bot_initial]}
    game.play_round(player_move, bot_move)

    # Draw the first card and select initial action using epsilon-greedy
    if len(game.deck.cards) == 0:
        continue
    player_card, bot_card = game.deal_next_cards()
    card = player_card[0]

    state = get_discrete_state(game.player_hand, card)
    valid_actions = get_valid_actions(game.player_hand)

    if not valid_actions:
        # no valid actions available, skip turn or break loop
        continue  # or handle gracefully

    if random.random() < EPSILON:
        action = random.choice(valid_actions)
    else:
        q_values = [Q[(state, a)] for a in valid_actions]
        max_q = max(q_values)
        best_actions = [a for a, q in zip(valid_actions, q_values) if q == max_q]
        action = random.choice(best_actions)


    while not game.game_over():
        chosen_pos = positions[action]
        move = {'cards': [card], 'positions': [(chosen_pos, card)]}
        bot_play = random_bot_agent(game.bot_hand, bot_card[0], game.player_hand)

        game.play_round(move, bot_play)

        # Prepare next step
        if len(game.deck.cards) == 0 or game.game_over():
            reward = 0
            done = True
            next_state = None
            next_action = None
        else:
            player_card, bot_card = game.deal_next_cards()
            card = player_card[0]
            next_state = get_discrete_state(game.player_hand, card)
            valid_actions = get_valid_actions(game.player_hand)

            if not valid_actions:
                next_action = None
            else:
                if random.random() < EPSILON:
                    next_action = random.choice(valid_actions)
                else:
                    q_values = [Q[(next_state, a)] for a in valid_actions]
                    next_action = valid_actions[q_values.index(max(q_values))]

            reward = 0
            done = False

        # If game ended, calculate final reward
        if done:
            player_score, _ = game.calculate_scores()
            reward = player_score

        # SARSA update rule:
        # Q(s,a) += alpha * (reward + gamma * Q(s',a') - Q(s,a))
        q_next = Q[(next_state, next_action)] if next_state is not None and next_action is not None else 0.0
        Q[(state, action)] += ALPHA * (reward + GAMMA * q_next - Q[(state, action)])

        if done:
            break

        # Update state and action for next iteration
        state = next_state
        action = next_action

print("Saving SARSA Q-table...")
with open("SARSA/sarsa_table.pkl", "wb") as f:
    pickle.dump(dict(Q), f)
print("Done.")