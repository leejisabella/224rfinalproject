import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import tensorflow as tf # type: ignore

from dqn_agent import DQNAgent
from dqn_env import DQNEnvironment

# Setup TensorBoard writer
log_dir = "logs"
writer = tf.summary.create_file_writer(log_dir)

# Setup
state_size = 27
action_size = 3
agent = DQNAgent(state_size, action_size)
env = DQNEnvironment()

# Training parameters
num_episodes = 10000
batch_size = 32
update_target_freq = 10
checkpoint_freq = 100
window_size = 100  # Rolling average window
all_rewards = []
average_rewards = []

os.makedirs("data", exist_ok=True)

# Training loop
for e in range(num_episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    total_reward = 0

    for time in range(13):  # 13 card steps per hand
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        if done:
            break

    all_rewards.append(total_reward)
    avg_reward = np.mean(all_rewards[-window_size:])
    average_rewards.append(avg_reward)

    print(f"Episode {e}/{num_episodes} — Total Reward: {total_reward:.2f}, Average Reward: {avg_reward:.2f}")

    with writer.as_default():
        tf.summary.scalar("Total Reward", total_reward, step=e)
        tf.summary.scalar(f"{window_size}-Episode Average Reward", avg_reward, step=e)

    if len(agent.memory) > batch_size:
        agent.replay(batch_size)

    if e % update_target_freq == 0:
        agent.update_target_model()

    if e % checkpoint_freq == 0:
        agent.model.save(f"data/dqn_model_episode_{e}.h5")

writer.flush()

with open('data/training_log.csv', 'w', newline='') as f:
    writer_csv = csv.writer(f)
    writer_csv.writerow(['Episode', 'Reward', 'AverageReward'])
    for i in range(len(all_rewards)):
        writer_csv.writerow([i, all_rewards[i], average_rewards[i]])

# Plot training performance
plt.figure(figsize=(10, 5))
plt.plot(all_rewards, label='Episode Reward')
plt.plot(average_rewards, label=f'{window_size}-Episode Average')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('DQN Training Performance')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("data/training_rewards.png")
plt.show()
