from ofcp_player import OpenFaceChinesePoker, random_bot_agent
from dqn_agent import DQNAgent, extract_state_features
from dqn_initial_placement import find_best_initial_placement_dqn
import numpy as np
from tqdm import tqdm
import torch
import random

EPISODES = 2000
STATE_DIM = 728

agent = DQNAgent(state_dim=STATE_DIM)

for ep in tqdm(range(EPISODES), desc="Training DQN"):
    game = OpenFaceChinesePoker()
    player_initial, bot_initial = game.initial_deal()

    best_initial_move = find_best_initial_placement_dqn(game, player_initial, agent)
    bot_move = {'cards': bot_initial, 'positions': [('bottom', card) for card in bot_initial]}
    game.play_round(best_initial_move, bot_move)

    while not game.game_over():
        player_card, bot_card = game.deal_next_cards()
        state = extract_state_features(game.player_hand, player_card[0])

        legal_actions = ['top', 'middle', 'bottom']
        valid_actions = [i for i, pos in enumerate(legal_actions)
                            if len(getattr(game.player_hand, pos)) < (3 if pos == 'top' else 5)]

        action = agent.act(state)
        if action not in valid_actions:
            action = random.choice(valid_actions)

        chosen_pos = legal_actions[action]
        player_move = {'cards': [player_card[0]], 'positions': [(chosen_pos, player_card[0])]}
        bot_move = random_bot_agent(game.bot_hand, bot_card[0], game.player_hand)

        game.play_round(player_move, bot_move)
        next_state = extract_state_features(game.player_hand, player_card[0])  # state after play
        done = game.game_over()

        if done:
            player_score, bot_score = game.calculate_scores()
            reward = player_score
            agent.remember(state, action, reward, next_state, True)
        else:
            agent.remember(state, action, 0, next_state, False)

        agent.replay()

torch.save(agent.q_net.state_dict(), "dqn_model.pt")