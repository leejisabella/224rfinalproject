import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import tensorflow as tf # type: ignore

from dqn_agent import DQNAgent
from dqn_env import DQNEnvironment

log_dir = "dueling_double_dqn/logs"
writer = tf.summary.create_file_writer(log_dir)

state_size = 27
action_size = 3
agent = DQNAgent(state_size, action_size)
env = DQNEnvironment()

num_episodes = 100
batch_size = 32
checkpoint_freq = 100
window_size = 50

all_rewards = []
average_rewards = []

os.makedirs("data", exist_ok=True)

for e in range(num_episodes):
    state = env.reset()
    total_reward = 0

    for _ in range(env.max_steps):
        valid_actions = env.get_valid_actions()
        action = agent.act(state, valid_actions)

        next_state, reward, done = env.step(action)

        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        if done:
            break

    all_rewards.append(total_reward)
    avg_reward = np.mean(all_rewards[-window_size:])
    average_rewards.append(avg_reward)

    print(f"Episode {e}/{num_episodes} — Reward: {total_reward:.2f}, Avg: {avg_reward:.2f}")

    with writer.as_default():
        tf.summary.scalar("Total Reward", total_reward, step=e)
        tf.summary.scalar(f"{window_size}-Episode Avg Reward", avg_reward, step=e)

    if len(agent.memory) >= batch_size:
        agent.replay(batch_size)

    if e % checkpoint_freq == 0:
        agent.model.save(f"data/dqn_model_ep{e}.h5")

writer.flush()

# Save training log
with open('data/training_log.csv', 'w', newline='') as f:
    writer_csv = csv.writer(f)
    writer_csv.writerow(['Episode', 'Reward', 'AverageReward'])
    for i in range(len(all_rewards)):
        writer_csv.writerow([i, all_rewards[i], average_rewards[i]])

# Plot
plt.figure(figsize=(10, 5))
plt.plot(all_rewards, label='Reward')
plt.plot(average_rewards, label=f'{window_size}-Episode Avg')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Dueling DQN Training')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("data/training_rewards.png")
plt.show()