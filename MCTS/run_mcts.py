from ofcp_player import OpenFaceChinesePoker, random_bot_agent
from mcts import MCTSAgent
from mcts_initial_placement import find_best_initial_placement
from copy import deepcopy

if __name__ == "__main__":
    game = OpenFaceChinesePoker()
    agent = MCTSAgent(num_simulations=50)

    player_initial, bot_initial = game.initial_deal()

    print("Initial Deals:")
    print(player_initial)
    print(bot_initial)

    # Use MCTS to find the optimal initial placement
    best_initial_move = find_best_initial_placement(game, player_initial)
    bot_move = {'cards': bot_initial, 'positions': [('bottom', card) for card in bot_initial]}

    game.play_round(best_initial_move, bot_move)

    while not game.game_over():
        player_card, bot_card = game.deal_next_cards()
        print("New cards:")
        print(player_card)
        print(bot_card)

        chosen_hand = agent.search(game, player_card[0])

        move = {'cards': [player_card[0]], 'positions': []}
        for pos in ['top', 'middle', 'bottom']:
            if getattr(chosen_hand, pos) != getattr(game.player_hand, pos):
                move['positions'].append((pos, player_card[0]))
                break

        bot_move = random_bot_agent(game.bot_hand, bot_card[0], game.player_hand)
        game.play_round(move, bot_move)
        print("Player Hand at end of round:")
        print(game.player_hand.top)
        print(game.player_hand.middle)
        print(game.player_hand.bottom)
        print("Bot Hand at end of round:")
        print(game.bot_hand.top)
        print(game.bot_hand.middle)
        print(game.bot_hand.bottom)
        print("")

    player_points, bot_points = game.calculate_scores()
    print("Final Scores:")
    print(f"Player (MCTS Agent): {player_points}")
    print(f"Bot (Random Agent): {bot_points}")