import numpy as np
from collections import deque
from keras.models import Sequential # type: ignore
from keras.layers import Dense, BatchNormalization, Input # type: ignore
from keras.optimizers import Adam # type: ignore

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size  # 27 card positions + current card
        self.action_size = action_size  # 3 (front/mid/back)
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        
    def _build_model(self):
        model = Sequential()
        model.add(Dense(256, input_dim=self.state_size, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(256, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(256, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size, m_simulations=10):
        minibatch = np.random.choice(len(self.memory), batch_size, replace=False)
        states, targets = [], []
        
        for idx in minibatch:
            state, action, reward, next_state, done = self.memory[idx]
            target = self.model.predict(state)
            
            if done:
                target[0][action] = reward
            else:
                q_futures = []
                for _ in range(m_simulations):
                    t = self.target_model.predict(next_state)[0]
                    q_futures.append(np.max(t))
                target[0][action] = reward + self.gamma * np.mean(q_futures)
            
            states.append(state[0])
            targets.append(target[0])
        
        self.model.fit(np.array(states), np.array(targets), epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
