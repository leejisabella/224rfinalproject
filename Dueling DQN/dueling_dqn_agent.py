import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

class DuelingQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DuelingQNetwork, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        x = self.feature(x)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        q_vals = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_vals

class DuelingDQNAgent:
    def __init__(self, state_dim, action_dim=3, gamma=0.99, lr=1e-3, batch_size=64, buffer_size=10000, tau=0.01):
        self.q_net = DuelingQNetwork(state_dim, action_dim)
        self.target_net = DuelingQNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.action_dim = action_dim

    def act(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return torch.argmax(q_values).item()

    def store(self, transition):
        self.replay_buffer.push(transition)

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        curr_q = self.q_net(states).gather(1, actions)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1, keepdim=True)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = nn.MSELoss()(curr_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update()

    def soft_update(self):
        for target_param, param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
    
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
        max_len = 3 if zone == 'top' else 5
        cards = getattr(player_hand, zone)
        for card in cards[:max_len]:
            state_vec.extend(encode_card(card))
        for _ in range(max_len - min(len(cards), max_len)):
            state_vec.extend([0] * 52)
    
    state_vec.extend(encode_card(current_card))

    return np.array(state_vec, dtype=np.float32)