import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from ofcp_player import OpenFaceChinesePoker, random_bot_agent


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

# Simplified state encoder
def encode_state(hand, card):
    state = [len(hand.top), len(hand.middle), len(hand.bottom)]
    ranks = '23456789TJQKA'
    suits = 'CDHS'
    state += [ranks.index(card.rank), suits.index(card.suit)]
    return torch.tensor(state, dtype=torch.float32)

def train():
    env = OpenFaceChinesePoker()
    state_dim = 5
    action_dim = 3
    agent = PPOAgent(state_dim, action_dim)
    optimizer = optim.Adam(agent.parameters(), lr=0.001)
    num_episodes = 500

    for episode in range(num_episodes):
        env = OpenFaceChinesePoker()
        done = False
        episode_rewards = []
        log_probs, values, rewards = [], [], []

        # Initial dealing
        player_cards, bot_cards = env.initial_deal()
        player_card = player_cards.pop()

        while not done:
            state = encode_state(env.player_hand, player_card)
            pi, value = agent(state)
            dist = torch.distributions.Categorical(pi)
            action = dist.sample()
            positions = ['top', 'middle', 'bottom']
            available_positions = [
                p for p in positions if len(getattr(env.player_hand, p)) < (3 if p == 'top' else 5)
            ]


            chosen_pos = positions[action.item()] if positions[action.item()] in available_positions else available_positions[0]
            player_move = {'cards': [player_card], 'positions': [(chosen_pos, player_card)]}
            bot_move = random_bot_agent(env.bot_hand, bot_cards.pop(), env.player_hand)
            env.play_round(player_move, bot_move)

            if not env.game_over():
                player_card, bot_card = env.deal_next_cards()
                player_card = player_card[0]
            else:
                done = True

            log_prob = dist.log_prob(action)
            log_probs.append(log_prob)
            values.append(value)

        player_score, bot_score = env.calculate_scores()
        reward = player_score - bot_score
        rewards = [reward] * len(log_probs)

        # PPO Update
        returns = torch.tensor(rewards)
        values = torch.cat(values)
        log_probs = torch.stack(log_probs)
        advantage = returns - values.squeeze()
        policy_loss = -(log_probs * advantage.detach()).mean()
        value_loss = advantage.pow(2).mean()
        loss = policy_loss + 0.5 * value_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'Episode {episode}, Reward: {reward}')

 
if __name__ == '__main__':
    train()