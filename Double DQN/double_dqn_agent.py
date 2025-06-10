import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, x):
        return self.layers(x)

class DoubleDQNAgent:
    def __init__(self, state_dim, action_dim=3, gamma=0.99, lr=1e-4, batch_size=64, memory_size=50000, tau=1e-3):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        self.q_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        self.memory = ReplayBuffer(memory_size)
        self.epsilon_scheduler = LinearEpsilonScheduler(start=1.0, end=0.1, decay_steps=1000)
        self.steps = 0

        self.update_target_network()

    def act(self, state):
        self.steps += 1
        epsilon = self.epsilon_scheduler.get_epsilon(self.steps)

        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)

        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_net(state)
            return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.push((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        dones = torch.BoolTensor(dones).unsqueeze(1)

        q_values = self.q_net(states).gather(1, actions)

        # target network for DQN
        with torch.no_grad():
            next_actions = self.q_net(next_states).argmax(1, keepdim=True)
            target_q_values = self.target_net(next_states).gather(1, next_actions)
            target = rewards + (1 - dones.float()) * self.gamma * target_q_values

        loss = self.criterion(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update()

    def soft_update(self):
        for target_param, param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def update_target_network(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

def extract_state_features(hand, incoming_card):
    positions = ['top', 'middle', 'bottom']
    state_vector = np.zeros(728)

    card_to_index = lambda c: "23456789TJQKA".index(c.rank) * 4 + "CDHS".index(c.suit)

    for i, pos in enumerate(positions):
        for card in getattr(hand, pos):
            idx = card_to_index(card)
            state_vector[i * 52 + idx] = 1

    state_vector[3 * 52 + card_to_index(incoming_card)] = 1
    return state_vector

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class LinearEpsilonScheduler:
    def __init__(self, start=1.0, end=0.1, decay_steps=1000):
        self.start = start
        self.end = end
        self.decay_steps = decay_steps

    def get_epsilon(self, step):
        return max(self.end, self.start - (self.start - self.end) * step / self.decay_steps)
