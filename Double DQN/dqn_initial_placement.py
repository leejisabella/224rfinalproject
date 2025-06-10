# extracted from mcts initial placement of cards

import itertools
from copy import deepcopy
import torch
from double_dqn_agent import extract_state_features

def get_all_valid_initial_placements(cards):
    positions = ['top', 'middle', 'bottom']
    max_slots = {'top': 3, 'middle': 5, 'bottom': 5}
    valid_placements = []

    for pos_combo in itertools.product(positions, repeat=5):
        counts = {'top': 0, 'middle': 0, 'bottom': 0}
        for pos in pos_combo:
            counts[pos] += 1
        if all(counts[pos] <= max_slots[pos] for pos in positions):
            move = {'cards': cards, 'positions': list(zip(pos_combo, cards))}
            valid_placements.append(move)
    return valid_placements

def evaluate_initial_placement_dqn(game_state, initial_move, dqn_agent):
    temp_game = deepcopy(game_state)
    temp_game.play_round(initial_move, {'cards': [], 'positions': []})

    if len(temp_game.deck.cards) == 0:
        return float('-inf')

    next_card = temp_game.deck.draw(1)[0]
    state = extract_state_features(temp_game.player_hand, next_card)

    with torch.no_grad():
        q_values = dqn_agent.q_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
    
    return max(q_values)

def find_best_initial_placement_dqn(game_state, cards, dqn_agent):
    all_placements = get_all_valid_initial_placements(cards)
    best_score = float('-inf')
    best_move = None

    for move in all_placements:
        score = evaluate_initial_placement_dqn(game_state, move, dqn_agent)
        if score > best_score:
            best_score = score
            best_move = move

    return best_move
