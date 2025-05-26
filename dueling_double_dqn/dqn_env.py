import numpy as np
from treys import Evaluator, Card # type: ignore

class DQNEnvironment:
    def __init__(self):
        self.state_size = 27  # 3 rows * 9 cards (max)
        self.max_steps = 13
        self.reset()
        self.current_step = 0
        self.state_size = 27
        self.deck = list(range(52))
        self.front_row = []
        self.middle_row = []
        self.back_row = []
        self.evaluator = Evaluator()


    def reset(self):
        self.front_row = []
        self.middle_row = []
        self.back_row = []
        self.current_step = 0
        self.deck = list(range(52))
        np.random.shuffle(self.deck)
        self.current_card = self.deck.pop()
        return self._get_state()

    def _get_state(self):
        state = np.zeros((self.state_size,))
        rows = [self.front_row, self.middle_row, self.back_row]
        for i, row in enumerate(rows):
            for j, card in enumerate(row):
                state[i * 9 + j] = card
        return state.reshape(1, -1)

    def legal_actions(self):
        actions = []
        if len(self.front_row) < 3:
            actions.append(0)
        if len(self.middle_row) < 5:
            actions.append(1)
        if len(self.back_row) < 5:
            actions.append(2)
        return actions

    def step(self, action):
        if self.current_step >= len(self.deck):
            return self._get_state(), -10.0, True  # deck exhausted

        card = self.deck[self.current_step]

        if action == 0 and len(self.front_row) < 3:
            self.front_row.append(card)
        elif action == 1 and len(self.middle_row) < 5:
            self.middle_row.append(card)
        elif action == 2 and len(self.back_row) < 5:
            self.back_row.append(card)
        else:
            return self._get_state(), -10.0, True  # illegal move

        self.current_step += 1
        done = self.current_step >= 13

        if done:
            reward = self._evaluate_hand()
        else:
            reward = 0.0

        return self._get_state(), reward, done

    def _evaluate_hand(self):
        if not (len(self.front_row) == 3 and len(self.middle_row) == 5 and len(self.back_row) == 5):
            return -10.0  # foul

        front_score = self._evaluate_3card_hand(self.front_row)

        middle = [Card.new(self._card_str(c)) for c in self.middle_row]
        back = [Card.new(self._card_str(c)) for c in self.back_row]

        middle_score = self.evaluator.evaluate([], middle)
        back_score = self.evaluator.evaluate([], back)

        if front_score > self._evaluate_3card_hand(self.middle_row) or middle_score < back_score:
            return -10.0  # rule violation

        # Normalize treys scores
        middle_reward = (7462 - middle_score) / 7462 * 1.5
        back_reward = (7462 - back_score) / 7462 * 1.5
        front_reward = front_score  # Already small value (1–3)

        return front_reward + middle_reward + back_reward


    def _card_str(self, card_id):
        # Map 0–51 to string format like 'As', 'Td', etc.
        ranks = '23456789TJQKA'
        suits = 'cdhs'
        rank = ranks[card_id % 13]
        suit = suits[card_id // 13]
        return rank + suit

    def _evaluate_3card_hand(self, cards):
        ranks = [c % 13 for c in cards]
        if len(set(ranks)) == 1:
            return 3  # Three of a kind
        elif len(set(ranks)) == 2:
            return 2  # Pair
        else:
            return 1  # High card
