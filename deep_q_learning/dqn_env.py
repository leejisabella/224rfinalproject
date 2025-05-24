import numpy as np

class DQNEnvironment:
    def __init__(self):
        self.current_step = 0
        self.state_size = 27
    
    def reset(self):
        self.current_step = 0
        return np.random.rand(self.state_size)
    
    def step(self, action):
        self.current_step += 1
        next_state = np.random.rand(self.state_size)
        reward = np.random.rand()
        done = self.current_step >= 13
        return next_state, reward, done