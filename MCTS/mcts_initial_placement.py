import itertools
from copy import deepcopy
from ofcp_player import OpenFaceChinesePoker, random_bot_agent
from mcts import MCTSAgent  # Assuming your MCTS implementation from above.
from tqdm import tqdm


def get_all_initial_placements(cards):
    positions = ['top', 'middle', 'bottom']
    max_slots = {'top': 3, 'middle': 5, 'bottom': 5}

    # Generate valid initial card placements
    valid_placements = []

    # Generate all possible position assignments for 5 cards considering the limits
    for placement in itertools.product(positions, repeat=5):
        placement_counts = {'top': 0, 'middle': 0, 'bottom': 0}
        for pos in placement:
            placement_counts[pos] += 1

        if all(placement_counts[pos] <= max_slots[pos] for pos in positions):
            valid_placements.append(placement)

    # Each placement maps cards to positions
    initial_moves = []
    for placement in valid_placements:
        moves = {'cards': [], 'positions': []}
        for pos, card in zip(placement, cards):
            moves['cards'].append(card)
            moves['positions'].append((pos, card))
        initial_moves.append(moves)

    return initial_moves


def evaluate_initial_placement(existing_game_state, initial_move, bot_cards, num_simulations=50):
    # Create a deep copy to avoid mutating original state
    game = deepcopy(existing_game_state)

    # Apply the initial placement under consideration
    bot_initial_move = random_bot_agent(game.bot_hand, bot_cards, game.player_hand)
    game.play_round(initial_move, bot_initial_move)

    mcts_agent = MCTSAgent(num_simulations=num_simulations)
    total_result = 0
    simulations = 5  # Run multiple rollouts for averaging results

    for _ in range(simulations):
        sim_game = deepcopy(game)
        
        # Continue simulation with MCTS moves
        while not sim_game.game_over():
            player_card, bot_card = sim_game.deal_next_cards()

            chosen_pos = mcts_agent.search(sim_game, player_card[0])
            if chosen_pos is None:
                # No legal actions; break the simulation early
                break
            move = {'cards': [player_card[0]], 'positions': [(chosen_pos, player_card[0])]}

            bot_move = random_bot_agent(sim_game.bot_hand, bot_card, sim_game.player_hand)
            sim_game.play_round(move, bot_move)

        player_points, bot_points = sim_game.calculate_scores()
        total_result += player_points

    avg_result = total_result / simulations
    return avg_result


def find_best_initial_placement(existing_game_state, cards, bot_cards):
    initial_placements = get_all_initial_placements(cards)
    best_move = None
    best_score = float('-inf')

    for move in tqdm(initial_placements):
        score = evaluate_initial_placement(existing_game_state, move, bot_cards)
        # print(f"Placement {move['positions']} evaluated with avg. score: {score}")
        if score > best_score:
            best_score = score
            best_move = move

    return best_move