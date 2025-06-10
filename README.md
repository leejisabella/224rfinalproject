# Advancing Multi-Agent Reasoning in Open-Face Chinese Poker

## Motivation

While reinforcement learning (RL) has been extensively studied in games like No-Limit Texas Hold’em poker, Open-Face Chinese Poker (OFCP) remains largely unexplored. OFCP presents unique challenges due to its **sparse reward system** and **complex hand dynamics**. This paper investigates RL methods of **Deep Q-Learning**, **Proximal Policy Optimization (PPO)**, and **Monte Carlo Tree Search (MCTS)** to see if it can be effective for OFCP.

## Methods

We implemented and compared several RL algorithms in a **self-play environment** with no external datasets, pitting our agents against both rule-based and learning-based opponents.

### Q-Learning Family

- **Q-Learning** Baseline method due to small state spaces.
- **Deep Q-Learning** Inspired by Tan and Xiao (2018) implementation of DQN for OFCP.
- **Double DQN**: reduces overestimation bias by decoupling action selection and evaluation.
- **Dueling DQN**: separates the network into value and advantage streams for more precise Q-value approximation.

### PPO (Proximal Policy Optimization)

- Stable policy gradient learning and stochastic decision-making, suitable for imperfect information and uncertainty.
- Includes clipped surrogate optimization, entropy regularization, and **Generalized Advantage Estimation (GAE)** for improved tie-breaking.

### MCTS (Monte Carlo Tree Search)

- Explores potential card placements and simulates future rollouts.
- Implements optimizations like **Cross-Entropy Method (CEM)**, **Rapid Action Value Estimation (RAVE)**, and **Counterfactual Regret Minimization (CFR)** to improve early move decisions and long-term planning.

## Implementation

- Custom two-player OFCP environment with deck management, card placement, and hand validation.
- Rewards aligned with OFCP rules (winning hands, fouling, scooping, royalties).
- Agents trained via self-play and evaluated by:
  - Method win rate
  - Bot win rate
  - Average points per game
  - Training efficiency

## Results

| Method     | Win Rate (Model) | Bot Win Rate | Avg Points/Game | Evaluation Time (100 games) |
| ---------- | ---------------- | ------------ | --------------- | --------------------------- |
| **MCTS**   | **89%**          | 3%           | 11.2            | 447 minutes                 |
| PPO + GAE  | 41%              | 0%           | 5.02            | 20 seconds                  |
| Double DQN | 35%              | 23%          | 3.50            | 4m 16s                      |

- **MCTS** outperformed other methods with the highest win rate and points, but incurred a significant computational cost (~200x slower than PPO).
- PPO demonstrated competitive performance with much faster evaluation.
- Q-learning methods were less effective overall but still outperformed random play.

## Discussion and Conclusion

- Our focus was on two-player OFCP, excluding multi-player variants and advanced rule sets (e.g., Fantasyland, Shoot the Moon).
- While **MCTS offers superior gameplay quality**, its computational overhead limits real-time applications.
- Future work aims to:
  - Optimize MCTS efficiency (e.g., parallel rollouts, learned policies).
  - Explore hybrid neuroevolution-RL techniques to handle sparse rewards.
  - Enhance lightweight methods for better decisiveness without heavy compute.

**Key takeaway:** All three classes of RL methods outperform a random bot, with MCTS showing the strongest performance (89% method win rate).
