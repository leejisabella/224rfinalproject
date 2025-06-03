from ofcp_player import OpenFaceChinesePoker, random_bot_agent
from mcts import MCTSAgent
from mcts_initial_placement import find_best_initial_placement
from copy import deepcopy
from tqdm import tqdm

if __name__ == "__main__":
    num_runs = 10

    total_p_points, total_b_points = 0, 0
    total_p_wins, total_b_wins = 0, 0

    for run in tqdm(range(num_runs)):
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
            # print("New cards:")
            # print(player_card)
            # print(bot_card)

            # Explicitly returns the chosen position (top, middle, or bottom)
            chosen_pos = agent.search(game, player_card[0])
            move = {'cards': [player_card[0]], 'positions': [(chosen_pos, player_card[0])]}

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

        total_p_points += player_points
        total_b_points += bot_points

        if player_points > bot_points:
            total_p_wins += 1
        if player_points < bot_points:
            total_b_wins += 1

    print(f"Player Total Points: {str(total_p_points)}")
    print(f"Bot Total Points: {str(total_b_points)}")

    print(f"Player Average Points: {str(total_p_points / num_runs)}")
    print(f"Bot Average Points: {str(total_b_points / num_runs)}")

    print(f"Player Num Wins: {str(total_p_wins)}")
    print(f"Player Winrate: {str(total_p_wins / num_runs)}")
    print(f"Bot Num Wins: {str(total_b_wins)}")