import random
import math
from copy import deepcopy
from ofcp_player import OpenFaceChinesePoker, random_bot_agent, Card

class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0
        self.untried_actions = self.get_legal_actions()

    def get_legal_actions(self):
        actions = []
        positions = ['top', 'middle', 'bottom']
        for pos in positions:
            if len(getattr(self.state.player_hand, pos)) < (3 if pos == 'top' else 5):
                actions.append(pos)
        return actions

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self, c_param=1.41):
        if not self.children:
            return None
        choices_weights = [
            (child.value / child.visits) + c_param * math.sqrt((2 * math.log(self.visits) / child.visits))
            for child in self.children
        ]
        return self.children[choices_weights.index(max(choices_weights))]


    def expand(self, card):
        action = self.untried_actions.pop()
        next_state = deepcopy(self.state)
        bot_card = next_state.deck.draw(1)[0]
        next_state.play_round({'cards': [card], 'positions': [(action, card)]},
                              random_bot_agent(next_state.bot_hand, bot_card, next_state.player_hand))
        child_node = Node(next_state, parent=self)
        self.children.append(child_node)
        return child_node

    def backpropagate(self, result):
        self.visits += 1
        self.value += result
        if self.parent:
            self.parent.backpropagate(result)


class MCTSAgent:
    def __init__(self, num_simulations=100):
        self.num_simulations = num_simulations

    def search(self, initial_state, card):
        root = Node(deepcopy(initial_state))

        for _ in range(self.num_simulations):
            node = root

            # Selection
            while not node.is_fully_expanded() and node.children:
                node = node.best_child()

            # Expansion
            if node.untried_actions:
                node = node.expand(card)

            # Simulation (Rollout)
            result = self.rollout(deepcopy(node.state))

            # Backpropagation
            node.backpropagate(result)

        best_child_node = root.best_child(c_param=0)
        if best_child_node is None:
            # Return original state if no child exists (no valid moves)
            return initial_state.player_hand

        return best_child_node.state.player_hand
    
    def rollout(self, state):
        while not state.game_over():
            player_card, bot_card = state.deal_next_cards()

            player_move = random_bot_agent(state.player_hand, player_card[0], state.bot_hand)
            bot_move = random_bot_agent(state.bot_hand, bot_card[0], state.player_hand)

            if player_move['positions'] and bot_move['positions']:
                state.play_round(player_move, bot_move)
            else:
                break

        player_points, bot_points = state.calculate_scores()  # Use new direct scoring method
        return player_points - bot_points
