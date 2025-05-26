import numpy as np
import random
from collections import deque
from keras.models import Model # type: ignore
from keras.layers import Input, Dense, Lambda, Subtract, Add # type: ignore
from keras.optimizers import Adam # type: ignore
import tensorflow.keras.backend as K # type: ignore

class DQNAgent:
    def __init__(self, state_size, action_size, gamma=0.99, epsilon=1.0,
                 epsilon_min=0.05, epsilon_decay=0.995, learning_rate=0.0005,
                 tau=0.1):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.tau = tau  # Soft update factor

        self.model = self._build_model()
        self.target_model = self._build_model()
        self._update_target_model(hard_update=True)

    def _build_model(self):
        inputs = Input(shape=(self.state_size,))
        x = Dense(256, activation='relu')(inputs)
        x = Dense(256, activation='relu')(x)

        # Dueling architecture
        value = Dense(1)(x)
        advantage = Dense(self.action_size)(x)
        advantage_mean = Lambda(lambda a: K.mean(a, axis=1, keepdims=True))(advantage)
        advantage = Subtract()([advantage, advantage_mean])
        q_values = Add()([value, advantage])

        model = Model(inputs=inputs, outputs=q_values)
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def _update_target_model(self, hard_update=False):
        main_weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        new_weights = []

        for mw, tw in zip(main_weights, target_weights):
            new_weights.append(mw if hard_update else self.tau * mw + (1 - self.tau) * tw)

        self.target_model.set_weights(new_weights)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def get_valid_actions(self):
        valid_actions = []
        if len(self.front_row) < 3:
            valid_actions.append(0)
        if len(self.middle_row) < 5:
            valid_actions.append(1)
        if len(self.back_row) < 5:
            valid_actions.append(2)
        return valid_actions

    def act(self, state, valid_actions=None):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(valid_actions)
        
        q_values = self.model.predict(state, verbose=0)[0]
        masked_q = np.full_like(q_values, -np.inf)
        for a in valid_actions:
            masked_q[a] = q_values[a]
        return np.argmax(masked_q)

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        states, targets = [], []

        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state, verbose=0)
            target_val = self.target_model.predict(next_state, verbose=0)

            if done:
                target[0][action] = reward
            else:
                best_action = np.argmax(self.model.predict(next_state, verbose=0)[0])
                target[0][action] = reward + self.gamma * target_val[0][best_action]

            states.append(state[0])
            targets.append(target[0])

        self.model.fit(np.array(states), np.array(targets), epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self._update_target_model()