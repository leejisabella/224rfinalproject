from ofcp_player import OpenFaceChinesePoker, random_bot_agent
from mcts import MCTSAgent
from mcts_initial_placement import find_best_initial_placement
from copy import deepcopy
from tqdm import tqdm
import numpy as np
import random
import argparse

def evaluate_agent(agent, num_games=5):
    total_points = 0

    for _ in range(num_games):
        game = OpenFaceChinesePoker()
        
        player_initial, bot_initial = game.initial_deal()
        
        # Optimal initial placement (using your existing method)
        best_initial_move = find_best_initial_placement(game, player_initial, bot_initial)
        bot_move = random_bot_agent(game.bot_hand, bot_initial, game.player_hand)
        game.play_round(best_initial_move, bot_move)

        # Play the game fully
        while not game.game_over():
            player_card, bot_card = game.deal_next_cards()

            chosen_pos = agent.search(game, player_card[0])

            # if chosen_pos is None:
            #     available_positions = [
            #         pos for pos in ['top', 'middle', 'bottom']
            #         if len(getattr(game.player_hand, pos)) < (3 if pos == 'top' else 5)
            #     ]
            # chosen_pos = random.choice(available_positions) if available_positions else 'bottom'  # fallback option

            move = {'cards': [player_card[0]], 'positions': [(chosen_pos, player_card[0])]}

            bot_move = random_bot_agent(game.bot_hand, bot_card, game.player_hand)
            game.play_round(move, bot_move)

        player_points, bot_points = game.calculate_scores()

        total_points += player_points  # clearly using score difference as metric

    return total_points / num_games  # average points per game


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cross_entropy", action='store_true')
    args = parser.parse_args()

    if args.cross_entropy:
        mean = 5   # initial guess for c_param
        std_dev = 2  # initial standard deviation
        num_samples = 5  # number of samples per iteration
        elite_fraction = 0.2  # top 20% elite samples
        num_iterations = 3  # total CEM iterations

        best_c_param = mean  # to track best parameter found

        for iteration in tqdm(range(num_iterations)):
            # Sample candidate parameters
            samples = np.random.normal(mean, std_dev, num_samples)

            # Evaluate each c_param
            performance = []
            for c_param in tqdm(samples, desc=f"Iteration {iteration+1}"):
                agent = MCTSAgent(num_simulations=100, c_param=c_param)
                total_reward = evaluate_agent(agent)  # define this clearly below
                performance.append((c_param, total_reward))

            # Select elite samples
            performance.sort(key=lambda x: x[1], reverse=True)  # sort by performance
            elite_cutoff = int(num_samples * elite_fraction)
            elite_samples = [x[0] for x in performance[:elite_cutoff]]

            # Update mean and std_dev
            mean = np.mean(elite_samples)
            std_dev = np.std(elite_samples)

            best_c_param = elite_samples[0]  # best parameter this iteration

            print(f"Iteration {iteration+1} | New mean: {mean:.4f}, New std_dev: {std_dev:.4f}, Best c_param: {best_c_param:.4f}")

        print(f"\nOptimal c_param found: {best_c_param:.4f}")

    else:
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
            bot_move = random_bot_agent(game.bot_hand, bot_initial, game.player_hand)

            game.play_round(best_initial_move, bot_move)

            while not game.game_over():
                player_card, bot_card = game.deal_next_cards()
                # print("New cards:")
                # print(player_card)
                # print(bot_card)

                # Explicitly returns the chosen position (top, middle, or bottom)
                chosen_pos = agent.search(game, player_card[0])
                move = {'cards': [player_card[0]], 'positions': [(chosen_pos, player_card[0])]}

                bot_move = random_bot_agent(game.bot_hand, bot_card, game.player_hand)
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