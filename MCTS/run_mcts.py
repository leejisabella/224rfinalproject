from ofcp_player import OpenFaceChinesePoker, random_bot_agent
from mcts import MCTSAgent
from mcts_initial_placement import find_best_initial_placement
from copy import deepcopy
from tqdm import tqdm
import numpy as np
import random
import argparse
from cfr import CFRPolicyBot, load_regret_table

def evaluate_agent(agent, num_games=5, rave=False, cfr=False):
    total_points = 0

    for _ in range(num_games):
        game = OpenFaceChinesePoker()
        
        player_initial, bot_initial = game.initial_deal()
        
        # Optimal initial placement (using your existing method)
        best_initial_move = find_best_initial_placement(game, player_initial, bot_initial, rave=rave, cfr=cfr)
        bot_move = random_bot_agent(game.bot_hand, bot_initial, game.player_hand)
        game.play_round(best_initial_move, bot_move)

        # Play the game fully
        while not game.game_over():
            player_card, bot_card = game.deal_next_cards()

            chosen_pos = agent.search(game, player_card[0], rave=args.rave)

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
    parser.add_argument("--log_file", type=str)
    parser.add_argument("--cross_entropy", action='store_true')
    parser.add_argument("--n_runs", type=int, default=10)
    parser.add_argument("--chosen_c_param", type=int, default=10.7251)
    parser.add_argument("--rave", action='store_true')
    parser.add_argument("--cfr", action='store_true')
    args = parser.parse_args()

    cfr_check = False
    if args.cfr:
        cfr_check = True

    if args.cross_entropy:
        with open(args.log_file, "w") as f:
            mean = 10.7251   # initial guess for c_param
            std_dev = 0.5  # initial standard deviation
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
                    agent = MCTSAgent(num_simulations=100, c_param=c_param, cfr=cfr_check)
                    total_reward = evaluate_agent(agent, rave=args.rave, cfr=cfr_check)  # define this clearly below
                    performance.append((c_param, total_reward))

                    f.write(f"Tracking: {c_param} parameter with {total_reward} reward. \n")

                # Select elite samples
                performance.sort(key=lambda x: x[1], reverse=True)  # sort by performance
                elite_cutoff = int(num_samples * elite_fraction)
                elite_samples = [x[0] for x in performance[:elite_cutoff]]

                # Update mean and std_dev
                mean = np.mean(elite_samples)
                std_dev = np.std(elite_samples)

                best_c_param = elite_samples[0]  # best parameter this iteration

                print(f"Iteration {iteration+1} | New mean: {mean:.4f}, New std_dev: {std_dev:.4f}, Best c_param: {best_c_param:.4f}")
                f.write(f"Iteration {iteration+1} | New mean: {mean:.4f}, New std_dev: {std_dev:.4f}, Best c_param: {best_c_param:.4f} \n")

            print(f"\nOptimal c_param found: {best_c_param:.4f}")
            f.write(f"\nOptimal c_param found: {best_c_param:.4f}")

    else:
        with open(args.log_file, "w") as f:
            num_runs = args.n_runs

            total_p_points, total_b_points = 0, 0
            total_p_wins, total_b_wins = 0, 0

            for run in tqdm(range(num_runs)):
                game = OpenFaceChinesePoker()

                if cfr_check:
                    load_regret_table("cfr_table.pkl")
                    cfr_bot = CFRPolicyBot()
                else:
                    cfr_bot = None

                agent = MCTSAgent(num_simulations=50, c_param=args.chosen_c_param, rollout_policy=cfr_bot, cfr=cfr_check)

                player_initial, bot_initial = game.initial_deal()

                print("Initial Deals:")
                print(player_initial)
                print(bot_initial)

                # Use MCTS to find the optimal initial placement
                best_initial_move = find_best_initial_placement(game, player_initial, bot_initial, rave=args.rave, cfr=cfr_check)
                bot_move = random_bot_agent(game.bot_hand, bot_initial, game.player_hand)

                game.play_round(best_initial_move, bot_move)

                while not game.game_over():
                    player_card, bot_card = game.deal_next_cards()
                    # print("New cards:")
                    # print(player_card)
                    # print(bot_card)

                    # Explicitly returns the chosen position (top, middle, or bottom)
                    chosen_pos = agent.search(game, player_card[0], rave=args.rave)
                    move = {'cards': [player_card[0]], 'positions': [(chosen_pos, player_card[0])]}

                    bot_move = random_bot_agent(game.bot_hand, bot_card, game.player_hand)
                    game.play_round(move, bot_move)

                    # print("Player Hand at end of round:")
                    # print(game.player_hand.top)
                    # print(game.player_hand.middle)
                    # print(game.player_hand.bottom)
                    # print("Bot Hand at end of round:")
                    # print(game.bot_hand.top)
                    # print(game.bot_hand.middle)
                    # print(game.bot_hand.bottom)
                    # print("")

                player_points, bot_points = game.calculate_scores()
                print("Final Scores:")
                print(f"Player (MCTS Agent): {player_points}")
                print(f"Bot (Random Agent): {bot_points}")

                f.write(f"Final Scores for Game {run}\n")
                f.write(f"Player (MCTS Agent): {player_points}\n")
                f.write(f"Bot (Random Agent): {bot_points}\n")

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

            # written results
            f.write("\n--- MCTS Summary ---\n")
            f.write(f"c_param: {args.chosen_c_param}\n")
            f.write(f"Total MCTS Points: {total_p_points}\n")
            f.write(f"Total Bot Points: {total_b_points}\n")
            f.write(f"Average MCTS Points: {total_p_points / num_runs:.2f}\n")
            f.write(f"Average Bot Points: {total_b_points / num_runs:.2f}\n")
            f.write(f"MCTS Wins: {total_p_wins}/{num_runs} ({100 * total_p_wins / num_runs:.1f}%)\n")
            f.write(f"Bot Wins: {total_b_wins}/{num_runs} ({100 * total_b_wins / num_runs:.1f}%)\n")

            print("\nEvaluation results saved to:", args.log_file)