import itertools
from copy import deepcopy
import pickle
import numpy as np

# Load SARSA Q-table
with open("Q-Learning/q_table_sarsa.pkl", "rb") as f:
    Q = pickle.load(f)

positions = ['top', 'middle', 'bottom']
max_slots = {'top': 3, 'middle': 5, 'bottom': 5}

def get_all_valid_initial_placements(cards):
    valid_placements = []
    for pos_combo in itertools.product(positions, repeat=5):
        counts = {'top': 0, 'middle': 0, 'bottom': 0}
        for pos in pos_combo:
            counts[pos] += 1
        if all(counts[pos] <= max_slots[pos] for pos in positions):
            move = {'cards': cards, 'positions': list(zip(pos_combo, cards))}
            valid_placements.append(move)
    return valid_placements

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
    valid = []
    for i, pos in enumerate(positions):
        if len(getattr(player_hand, pos)) < max_slots[pos]:
            valid.append(i)
    return valid

def evaluate_initial_placement_q(game_state, initial_move):
    temp_game = deepcopy(game_state)
    temp_game.play_round(initial_move, {'cards': [], 'positions': []})

    if len(temp_game.deck.cards) == 0:
        return float('-inf')

    next_card = temp_game.deck.draw(1)[0]
    state = get_discrete_state(temp_game.player_hand, next_card)
    valid_actions = get_valid_actions(temp_game.player_hand)

    # Lookup max Q value for next state using SARSA Q-table
    q_values = [Q.get((state, a), 0.0) for a in valid_actions]
    return max(q_values, default=0.0)

def find_best_initial_placement_q(game_state, cards):
    all_placements = get_all_valid_initial_placements(cards)
    best_score = float('-inf')
    best_move = None

    for move in all_placements:
        score = evaluate_initial_placement_q(game_state, move)
        if score > best_score:
            best_score = score
            best_move = move

    return best_move