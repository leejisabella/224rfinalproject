import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from MCTS.ofcp_player import OpenFaceChinesePoker, random_bot_agent, evaluate_hand

class PPOAgent(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc_pi = nn.Linear(128, output_dim)
        self.fc_v = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        pi = F.softmax(self.fc_pi(x), dim=-1)
        v = self.fc_v(x)
        return pi, v

# Richer state encoder: include pile strengths and card features
def encode_state(hand, card):
    def pile_features(pile):
        if pile:
            strength = evaluate_hand(pile)
            cat = strength[0]
            val = max(strength[1]) if isinstance(strength[1], list) else strength[1]
        else:
            cat, val = 0, 0
        return [cat, val]

    top_feats = pile_features(hand.top)
    mid_feats = pile_features(hand.middle)
    bot_feats = pile_features(hand.bottom)
    ranks = '23456789TJQKA'
    suits = 'CDHS'
    card_feat = [ranks.index(card.rank), suits.index(card.suit)]
    return torch.tensor(top_feats + mid_feats + bot_feats + card_feat, dtype=torch.float32)

# PPO training with score-difference rewards, entropy bonus, and foul penalties
def ppo_train(num_episodes=500, lr=0.0005):
    state_dim = 8    # 6 pile-strength features + 2 card features
    action_dim = 3
    agent = PPOAgent(state_dim, action_dim)
    optimizer = optim.Adam(agent.parameters(), lr=lr)

    for episode in range(num_episodes):
        env = OpenFaceChinesePoker()
        log_probs, values, rewards, states = [], [], [], []

        # Initial deal
        player_cards, bot_cards = env.initial_deal()
        player_card = player_cards.pop()
        bot_card = bot_cards.pop()
        done = False

        prev_p, prev_b = env.calculate_scores()
        prev_diff = prev_p - prev_b

        while not done:
            state = encode_state(env.player_hand, player_card)
            states.append(state)
            pi, value = agent(state)

            # Determine valid actions
            positions = ['top', 'middle', 'bottom']
            available_positions = [p for p in positions if len(getattr(env.player_hand, p)) < (3 if p == 'top' else 5)]
            if not available_positions:
                # No legal move: small penalty and end
                rewards.append(torch.tensor(-2.0))
                log_probs.append(torch.tensor(0.0))
                values.append(torch.tensor(0.0))
                break

            pos_to_idx = {'top': 0, 'middle': 1, 'bottom': 2}
            available_indices = [pos_to_idx[p] for p in available_positions]
            masked_pi = pi[available_indices].clone()
            if masked_pi.sum().item() == 0:
                # If all probabilities zero, use uniform
                masked_pi = torch.ones_like(masked_pi) / masked_pi.size(0)
            else:
                masked_pi = masked_pi / masked_pi.sum()

            dist = torch.distributions.Categorical(masked_pi)
            action = dist.sample()
            chosen_pos = available_positions[action.item()]
            log_prob = dist.log_prob(action)

            # Play round
            player_move = {'cards': [player_card], 'positions': [(chosen_pos, player_card)]}
            bot_move = random_bot_agent(env.bot_hand, bot_card, env.player_hand)
            env.play_round(player_move, bot_move)

            # Immediate partial-hand foul check
            if not env.player_hand.valid_hand() and not env.player_hand.is_complete():
                rewards.append(torch.tensor(-6.0))
                log_probs.append(log_prob)
                values.append(value)
                done = True
                break

            # If game complete, check final foul or score-difference
            if env.game_over():
                new_p, new_b = env.calculate_scores()
                if not env.player_hand.valid_hand():
                    rewards.append(torch.tensor(-6.0))
                else:
                    new_diff = new_p - new_b
                    rewards.append(torch.tensor(new_diff - prev_diff))
                    prev_diff = new_diff
                break

            # Normal step: rewards = Δ(score difference)
            new_p, new_b = env.calculate_scores()
            new_diff = new_p - new_b
            rewards.append(torch.tensor(new_diff - prev_diff))
            prev_diff = new_diff
            log_probs.append(log_prob)
            values.append(value)

            # Next cards
            next_cards = env.deal_next_cards()
            if not next_cards or not next_cards[0] or not next_cards[1]:
                break
            player_card, bot_card = next_cards[0][0], next_cards[1][0]

        # Ensure there's at least one reward
        if not rewards:
            rewards = [torch.tensor(-2.0)]
            log_probs = [torch.tensor(0.0)]
            values = [torch.tensor(0.0)]

        # Compute PPO losses
        returns = torch.tensor(rewards, dtype=torch.float32)
        values_tensor = torch.cat(values).squeeze()
        log_probs_tensor = torch.stack(log_probs)
        advantage = returns - values_tensor
        if advantage.size(0) > 1:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        else:
            advantage = advantage * 0  # zero advantage if single step

        policy_loss = -(log_probs_tensor * advantage.detach()).mean()
        value_loss = advantage.pow(2).mean()

        # Recompute entropy from stored states
        if states:
            state_batch = torch.stack(states)
            _, pis = agent(state_batch)
            entropy_batch = torch.distributions.Categorical(pis).entropy()
            mean_entropy = entropy_batch.mean()
        else:
            mean_entropy = torch.tensor(0.0)

        loss = policy_loss + 0.5 * value_loss - 0.01 * mean_entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if episode % 50 == 0:
            print(f"[Episode {episode}] Last Δ(total diff) = {prev_diff:.2f}, Sum(rewards) = {returns.sum():.2f}")

    return agent

# Evaluation remains unchanged
def evaluate(agent, num_games=10):
    total_player_score = 0
    total_bot_score = 0
    player_wins = 0
    bot_wins = 0

    for i in range(num_games):
        env = OpenFaceChinesePoker()
        player_cards, bot_cards = env.initial_deal()
        player_card = player_cards.pop()
        bot_card = bot_cards.pop()
        done = False

        while not done:
            state = encode_state(env.player_hand, player_card)
            pi, _ = agent(state)

            positions = ['top', 'middle', 'bottom']
            available_positions = [p for p in positions if len(getattr(env.player_hand, p)) < (3 if p == 'top' else 5)]
            if not available_positions:
                break

            pos_to_idx = {'top': 0, 'middle': 1, 'bottom': 2}
            available_indices = [pos_to_idx[p] for p in available_positions]
            masked_pi = pi[available_indices]
            if masked_pi.sum().item() == 0:
                masked_pi = torch.ones_like(masked_pi) / masked_pi.size(0)
            else:
                masked_pi = masked_pi / masked_pi.sum()

            dist = torch.distributions.Categorical(masked_pi)
            action = dist.sample()
            chosen_pos = available_positions[action.item()]

            player_move = {'cards': [player_card], 'positions': [(chosen_pos, player_card)]}
            bot_move = random_bot_agent(env.bot_hand, bot_card, env.player_hand)
            env.play_round(player_move, bot_move)

            if env.game_over():
                break

            next_cards = env.deal_next_cards()
            if not next_cards or not next_cards[0] or not next_cards[1]:
                break
            player_card = next_cards[0][0]
            bot_card = next_cards[1][0]

        p_score, b_score = env.calculate_scores()
        total_player_score += p_score
        total_bot_score += b_score
        if p_score > b_score:
            player_wins += 1
        elif b_score > p_score:
            bot_wins += 1

        print(f"Game {i+1}: Player = {p_score}, Bot = {b_score}")

    print("\n==== Evaluation Summary ====")
    print(f"Player Wins: {player_wins}")
    print(f"Bot Wins: {bot_wins}")
    print(f"Average Player Score: {total_player_score / num_games:.2f}")
    print(f"Average Bot Score: {total_bot_score / num_games:.2f}")

if __name__ == '__main__':
    trained_agent = ppo_train(num_episodes=2000)
    evaluate(trained_agent, num_games=100)
