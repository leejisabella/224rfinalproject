from ofcp_player import OpenFaceChinesePoker, random_bot_agent
from double_dqn_agent import DoubleDQNAgent, extract_state_features
from dqn_initial_placement import find_best_initial_placement_dqn
import torch
import numpy as np
from tqdm import tqdm

STATE_DIM = 728
NUM_GAMES = 100
MODEL_PATH = "Double DQN/results/double_dqn_model.pt"

def evaluate_double_dqn(agent, num_games=NUM_GAMES, log_file="Double DQN/results/double_dqn_eval_results.txt"):
    total_p_points = 0
    total_b_points = 0
    total_p_wins = 0
    total_b_wins = 0

    with open(log_file, "w") as f:
        for game_idx in tqdm(range(num_games), desc="Evaluating"):
            game = OpenFaceChinesePoker()

            player_initial, bot_initial = game.initial_deal()
            best_initial_move = find_best_initial_placement_dqn(game, player_initial, agent)
            bot_move = {'cards': bot_initial, 'positions': [('bottom', card) for card in bot_initial]}
            game.play_round(best_initial_move, bot_move)

            while not game.game_over():
                player_card, bot_card = game.deal_next_cards()

                state = extract_state_features(game.player_hand, player_card[0])
                legal_actions = ['top', 'middle', 'bottom']
                valid_actions = [i for i, pos in enumerate(legal_actions)
                                 if len(getattr(game.player_hand, pos)) < (3 if pos == 'top' else 5)]

                action = agent.act(state)
                if action not in valid_actions:
                    action = np.random.choice(valid_actions)

                chosen_pos = legal_actions[action]
                player_move = {'cards': [player_card[0]], 'positions': [(chosen_pos, player_card[0])]}
                bot_move = random_bot_agent(game.bot_hand, bot_card[0], game.player_hand)

                game.play_round(player_move, bot_move)

            player_points, bot_points = game.calculate_scores()
            total_p_points += player_points
            total_b_points += bot_points

            if player_points > bot_points:
                total_p_wins += 1
            elif bot_points > player_points:
                total_b_wins += 1

            f.write(f"Game {game_idx+1} | Double DQN: {player_points}, Bot: {bot_points}\n")

        f.write("\n--- Double DQN Evaluation Summary ---\n")
        f.write(f"Total Double DQN Points: {total_p_points}\n")
        f.write(f"Total Bot Points: {total_b_points}\n")
        f.write(f"Average Double DQN Points: {total_p_points / num_games:.2f}\n")
        f.write(f"Average Bot Points: {total_b_points / num_games:.2f}\n")
        f.write(f"Double DQN Wins: {total_p_wins}/{num_games} ({100 * total_p_wins / num_games:.1f}%)\n")
        f.write(f"Bot Wins: {total_b_wins}/{num_games} ({100 * total_b_wins / num_games:.1f}%)\n")

        print("\nEvaluation results saved to:", log_file)

if __name__ == "__main__":
    agent = DoubleDQNAgent(state_dim=STATE_DIM)
    agent.q_net.load_state_dict(torch.load(MODEL_PATH))
    agent.q_net.eval()
    evaluate_double_dqn(agent)
