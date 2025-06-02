from math import sqrt, log
from copy import deepcopy
from rng import rng
from tqdm import trange
import pickle
import numpy as np

from baselines import random_choice, highest_card
from game import Game


HEURISTIC = highest_card
N_EPISODES = 500
N_EXPLORATIONS_PER_TREE = 10

class Node:
    def __init__(self, game, finished, parent, action, c=0.3):
        self.children = None  # Dictionary of child nodes
        self.Q = 0  # Sum of rewards of rollouts from this node
        self.N = 0  # Number of visits to this node

        self.game = game  # Copy of the game environment so we can simulate it
        self.finished = finished  # whether the game is finished or not, so search stops
        self.parent = parent  # for backpropagation

        self.action = action # action we took to get here

        self.c = c  # Tunable parameter 


    def get_UCB_score(self):
        """ 
        Gives values to the node, and MCTS picks nodes with highest value. 
        As according to the textbook, algorithm takes the action that maximizes:
            Q(h, a) + c * sqrt(log(N(h))/N(h,a))
        where N(h) = total visit count for all children, N(h, a) is the visit count for the given action, 
        and c is an exploration parameter.
        """

        # Unexplored states have maximum value (favor exploration)
        if self.N == 0:
            return float("inf")
        
        # Need parent node of the current node
        top_node = self
        if self.parent:
            top_node = top_node.parent

        return (self.Q / self.N) + self.c * sqrt(log(top_node.N) / self.N)


    def create_child(self):
        """ 
        Child node for each possible action from this state. Apply this action to
        a copy of the current node environment and create a child node with that
        action applied
        """

        if self.finished:
            return
        
        actions = []
        games = []

        for c in self.game.player.hand:
            actions.append(c)
            new_game = deepcopy(self.game)
            games.append(new_game)

        children = {}
        #print(f"{actions=}")
        for action, game in zip(actions,games):
            (reward, done) = game.one_step(HEURISTIC, action=action) 
            children[action] = Node(game=game, finished=done, parent=self, action=action)

        self.children = children


    def explore(self):
        """
        Search along tree as follows:
        1. From the current node, continuously (recursively) pick the children which maximize the value 
        2. When a leaf is reached:
            a. If it has never been explored before, do a rollout and update current value
            b. Otherwise, expand the node and create its children. Pick a random child, and do a rollout update
        3. Backpropagate the updated statistics up the tree until the root: update both value and visit counts
        """ 
        # pudb.set_trace()
        current = self
        
        # Step 1: Keep picking children until we reach a leaf
        while current.children:
            children = current.children
            max_score = max(c.get_UCB_score() for c in children.values())
            actions = [a for a, c in children.items() if c.get_UCB_score() == max_score]
            if len(actions) == 0:
                print("There are no actions available at this score. ", max_score)
            action = rng.choice(actions)
            current = children[action]

        # Step 2: If the leaf has been explored before, expand node and pick a random child. Do a rollout
        if current.N >= 1:
            current.create_child()
            if current.children:
                #print(f"{current.children=}")
                current = rng.choice(list(current.children.values()))
        current.Q += current.rollout()
        current.N += 1

        # Step 3: Backpropagate up the tree until the root
        parent = current
        while parent.parent:
            parent = parent.parent
            parent.N += 1 
            parent.Q += current.Q


    def rollout(self):
        """
        Rollout is a random play from a copy of the environment of the current node.  
        Gives value for the current node, which seems random alone. But as number of
        rollouts increase, the more accurate the average of the value.

        So I think the random rollout should just be playing the game with the random strategy passed in.
        """ 

        if self.finished: 
            return 0
    
        new_game = deepcopy(self.game)
        #print(f"\n-----NEW ROLLOUT------")
        v = new_game.main(HEURISTIC)
        return v


    def next(self):
        """
        Once search is done, this function defines how to choose the next best action from our current node
        We choose the next action based on the highest Q value, which shows higher demonstrated success
        as a result of choosing that action next.
        """

        if self.finished:
            raise ValueError("Something is wrong. This game is supposed to be over. Stop playing.")
        if not self.children:
            raise ValueError("Something is wrong. You have no children. The game is not over.")
        
        # Pick the node that has the highest value Q
        max_val = 0
        max_child = None
        max_action = None

        #print(f"{self.children.items()=}")
        for action, node in self.children.items():
            if action in self.game.player.hand:
                val = node.N
                if val > max_val:
                    max_val = val
                    max_child = node
                    max_action = action

        return max_child, max_action


def policy_player_MCTS(my_tree, count):
    for i in range(count):
        my_tree.explore()

    next_tree, next_action = my_tree.next()
    # next_tree.detach_parent()

    return next_tree, next_action

def mean(lst):
    return sum(lst) / len(lst)


def main(explorations=200, c=0.3):
    rewards = []
    moving_average = []
    for e in trange(N_EPISODES):
    #for e in range(10):


        reward_e = 0    
        game = Game()
        done = False
        
        new_game = deepcopy(game)
        mytree = Node(game=new_game, finished=False, parent=None, action=None, c=c)
        
        #print('episode #' + str(e+1))
        
        while not done:
        
            mytree, action = policy_player_MCTS(mytree, explorations)
            
            (reward, done) = game.one_step(HEURISTIC, action=action) 
            #print(f"{done=}")
                            
            reward_e = reward_e + reward
            
            #game.render() # uncomment this if you want to see your agent in action!
                    
            if done:
                #print('reward_e ' + str(reward_e))
                break
            
        rewards.append(reward_e)
        moving_average.append(mean(rewards[-100:]))

    print(f"Moving Average of E={explorations}, C={c} is {moving_average[-1]}")

    print(f"Average of E={explorations}, C={c} is {sum(rewards)/len(rewards)}")

    print(f"Win rate of E={explorations}, C={c} is {sum([1 for x in rewards if x == 10_000]) / len(rewards)}")

    filepath = f"rewards_{explorations=}_{c=}_{N_EPISODES=}.npy"
    np.save(filepath, rewards, allow_pickle=True)

    
        

if __name__ == "__main__":
    #print("Explorations Test:")
    #main(explorations=100, c=0.3)
    #main(explorations=200, c=0.3)
    #main(explorations=300, c=0.3)

    #print("")

    # print("c Test, E = 100")
    # main(explorations=100, c=0.1)
    # main(explorations=100, c=0.2)
    # main(explorations=100, c=0.3)

    print("")

    # print("c Test, E = 200")
    # main(explorations=200, c=0.1)
    # main(explorations=200, c=0.2)
    # main(explorations=200, c=0.3)

    print("c Test, E = 100, c=0.1")
    main(explorations=100, c=0.1)