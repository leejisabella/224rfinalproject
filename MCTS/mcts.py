import random
import math
from copy import deepcopy
from ofcp_player import OpenFaceChinesePoker, random_bot_agent, Card

class Node:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action  # Explicitly store the chosen action (position)
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

    def best_child(self, c_param=0.3):
        if not self.children:
            return None
        choices_weights = [
            (child.value / child.visits) + c_param * math.sqrt((2 * math.log(self.visits) / child.visits))
            for child in self.children
        ]
        return self.children[choices_weights.index(max(choices_weights))]


    def expand(self, card):
        action = random.choice(self.untried_actions)
        self.untried_actions.remove(action)
        next_state = deepcopy(self.state)
        bot_card = next_state.deck.draw(1)[0]

        next_state.play_round(
            {'cards': [card], 'positions': [(action, card)]},
            random_bot_agent(next_state.bot_hand, bot_card, next_state.player_hand)
        )

        child_node = Node(next_state, parent=self, action=action)
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

        # Check immediately if there are no legal actions
        if not root.untried_actions:
            return None  # Clearly indicate no actions are available
        
        # Explicitly expand all possible actions at least once at the root
        while root.untried_actions:
            child_node = root.expand(card)
            result = self.rollout(deepcopy(child_node.state))
            child_node.backpropagate(result)

        # Continue running MCTS
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

        # print("Best child node")
        # print(best_child_node.action)
        # print("Legal Actions")
        # print(root.get_legal_actions())

        return best_child_node.action if best_child_node else random.choice(root.get_legal_actions())
    
    def rollout(self, state):
        first_move = True
        while not state.game_over():
            player_card, bot_card = state.deal_next_cards()

            if first_move:
                best_score = float('-inf')
                best_move = None

                # Test each legal position explicitly once per rollout
                for pos in ['top', 'middle', 'bottom']:
                    if len(getattr(state.player_hand, pos)) < (3 if pos == 'top' else 5):
                        sim_state = deepcopy(state)
                        sim_state.play_round(
                            {'cards': [player_card[0]], 'positions': [(pos, player_card[0])]},
                            random_bot_agent(sim_state.bot_hand, bot_card[0], sim_state.player_hand)
                        )
                        score, _ = sim_state.calculate_scores()
                        if score > best_score:
                            best_score = score
                            best_move = {'cards': [player_card[0]], 'positions': [(pos, player_card[0])]}

                player_move = best_move if best_move else random_bot_agent(state.player_hand, player_card[0], state.bot_hand)
                first_move = False
            else:
                player_move = random_bot_agent(state.player_hand, player_card[0], state.bot_hand)

            bot_move = random_bot_agent(state.bot_hand, bot_card[0], state.player_hand)

            if player_move['positions'] and bot_move['positions']:
                state.play_round(player_move, bot_move)
            else:
                break

        player_points, bot_points = state.calculate_scores()
        return player_points

