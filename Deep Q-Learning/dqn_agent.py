import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()

        # defaulting to this neural network
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim=3, lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.q_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        
        # could attempt to get different optimizations over here
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.replay_buffer = deque(maxlen=10000)
        self.batch_size = 64
        self.update_target_steps = 100
        self.step_counter = 0

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_net(state_tensor)
            return torch.argmax(q_values).item()

    # create memory buffer
    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # sample actions randomly in DQN
        batch = random.sample(self.replay_buffer, self.batch_size)

        # unzipping values
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        dones = torch.BoolTensor(dones).unsqueeze(1)

        q_values = self.q_net(states).gather(1, actions)
        next_q_values = self.target_net(next_states).max(1, keepdim=True)[0]
        target_q_values = rewards + (1 - dones.float()) * self.gamma * next_q_values

        loss = self.loss_fn(q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_counter += 1
        if self.step_counter % self.update_target_steps == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def extract_state_features(player_hand, current_card):
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
        if card is not None:
            rank_idx = rank_to_index.get(card.rank)
            suit_idx = suit_to_index.get(card.suit)
            if rank_idx is None or suit_idx is None:
                raise ValueError(f"Unexpected card: rank={card.rank}, suit={card.suit}")
            index = suit_idx * 13 + rank_idx
            vec[index] = 1
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

    return np.array(state_vec, dtype=np.float32)