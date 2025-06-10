from ofcp_player import OpenFaceChinesePoker, random_bot_agent
from dueling_dqn_agent import DuelingDQNAgent, extract_state_features
from dqn_initial_placement import find_best_initial_placement_dqn
import torch
import numpy as np
from tqdm import tqdm
import random

STATE_DIM = 728
NUM_EPISODES = 2000
SAVE_PATH = "Dueling DQN/results/dueling_dqn_model.pt"

agent = DuelingDQNAgent(state_dim=STATE_DIM)

for episode in tqdm(range(NUM_EPISODES), desc="Training Dueling DQN"):
    game = OpenFaceChinesePoker()
    player_initial, bot_initial = game.initial_deal()

    best_initial_move = find_best_initial_placement_dqn(game, player_initial, agent)
    bot_move = {'cards': bot_initial, 'positions': [('bottom', card) for card in bot_initial]}
    game.play_round(best_initial_move, bot_move)

    transitions = []

    while not game.game_over():
        if len(game.deck.cards) == 0:
            break
        player_card, bot_card = game.deal_next_cards()
        state = extract_state_features(game.player_hand, player_card[0])

        legal_actions = ['top', 'middle', 'bottom']
        valid_actions = [i for i, pos in enumerate(legal_actions)
                         if len(getattr(game.player_hand, pos)) < (3 if pos == 'top' else 5)]

        action = agent.act(state, epsilon=0.1)
        if action not in valid_actions:
            action = random.choice(valid_actions)

        chosen_pos = legal_actions[action]
        player_move = {'cards': [player_card[0]], 'positions': [(chosen_pos, player_card[0])]}
        bot_move = random_bot_agent(game.bot_hand, bot_card[0], game.player_hand)

        game.play_round(player_move, bot_move)

        done = game.game_over()
        next_state = extract_state_features(game.player_hand, player_card[0]) if not done else np.zeros(STATE_DIM)
        transitions.append((state, action, next_state, done))

    if game.game_over():
        player_score, bot_score = game.calculate_scores()
        final_reward = player_score

        for state, action, next_state, done in transitions:
            agent.store((state, action, final_reward, next_state, float(done)))
            agent.update()

    if (episode + 1) % 100 == 0:
        torch.save(agent.q_net.state_dict(), SAVE_PATH)
        print(f"Model saved at episode {episode + 1}")