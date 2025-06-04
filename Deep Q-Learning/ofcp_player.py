import random
from collections import namedtuple, Counter
from itertools import combinations

# Basic card representation
Card = namedtuple('Card', ['rank', 'suit'])

# Poker hand rankings helper
def evaluate_hand(cards):
    ranks = '23456789TJQKA'
    rank_values = {r: i for i, r in enumerate(ranks, 2)}

    def hand_rank(cards):
        counts = Counter(card.rank for card in cards)
        suits = [card.suit for card in cards]
        rank_counts = sorted(counts.values(), reverse=True)

        if len(cards) == 1:
            return (0, sorted([rank_values[card.rank] for card in cards], reverse=True))

        if len(cards) == 2:
            # Pair
            if rank_counts == [2]:
                return (1, rank_values[max(counts, key=lambda k: (counts[k], rank_values[k]))])
            # High card
            return (0, sorted([rank_values[card.rank] for card in cards], reverse=True))

        if len(cards) == 3:
            # 3 of a kind
            if rank_counts == [3]:
                return (3, rank_values[max(counts, key=counts.get)])
            # Pair
            if rank_counts == [2, 1]:
                return (1, rank_values[max(counts, key=lambda k: (counts[k], rank_values[k]))])
            # High card
            return (0, sorted([rank_values[card.rank] for card in cards], reverse=True))
        
        if len(cards) == 4:
            # 3 of a kind
            if rank_counts == [3, 1]:
                return (3, rank_values[max(counts, key=counts.get)])
            # 2 Pair
            if rank_counts == [2, 2]:
                pairs = sorted([rank_values[r] for r, c in counts.items() if c == 2], reverse=True)
                return (2, pairs)
            # Pair
            if rank_counts == [2, 1, 1]:
                return (1, rank_values[max(counts, key=counts.get)])
            # High card
            return (0, sorted([rank_values[card.rank] for card in cards], reverse=True))

        is_flush = (len(set(suits)) == 1 and len(cards) == 5)
        sorted_ranks = sorted([rank_values[card.rank] for card in cards])
        is_straight = (sorted_ranks == list(range(sorted_ranks[0], sorted_ranks[0] + 5)) and len(cards) == 5)

        if is_straight and is_flush:
            return (8, sorted_ranks[-1])
        if rank_counts == [4, 1]:
            return (7, rank_values[max(counts, key=counts.get)])
        if rank_counts == [3, 2]:
            return (6, rank_values[max(counts, key=counts.get)])
        if is_flush:
            return (5, sorted_ranks[::-1])
        if is_straight:
            return (4, sorted_ranks[-1])
        # 3 of a kind
        if rank_counts == [3, 1, 1]:
            return (3, rank_values[max(counts, key=counts.get)])
        # 2 Pair
        if rank_counts == [2, 2, 1]:
            pairs = sorted([rank_values[r] for r, c in counts.items() if c == 2], reverse=True)
            return (2, pairs)
        # Pair
        if rank_counts == [2, 1, 1, 1]:
            return (1, rank_values[max(counts, key=counts.get)])
        # High card
        return (0, sorted_ranks[::-1])

    return hand_rank(cards)

# Returns 1 if hand 1 is better, 2 if hand 2 is better, 0 if tie
def compare_hands(hand_1, hand_2):
    if hand_1 > hand_2:
        return 1
    elif hand_1 < hand_2:
        return 2
    else:
        return 0

# Helper function to calculate royalties
def calculate_royalties(hand, position):
    ranks = '23456789TJQKA'
    rank_values = {r: i for i, r in enumerate(ranks, 2)}

    top_royalties = {'66': 1, '77': 2, '88': 3, '99': 4, 'TT': 5, 'JJ': 6, 'QQ': 7, 'KK': 8, 'AA': 9,
                     '222': 10, '333': 11, '444': 12, '555': 13, '666': 14, '777': 15, '888': 16,
                     '999': 17, 'TTT': 18, 'JJJ': 19, 'QQQ': 20, 'KKK': 21, 'AAA': 22}

    middle_bottom_royalties = {
        'middle': {'Three of a kind': 2, 'Straight': 4, 'Flush': 8, 'Full house': 12,
                   'Four of a kind': 20, 'Straight flush': 30, 'Royal flush': 50},
        'bottom': {'Three of a kind': 0, 'Straight': 2, 'Flush': 4, 'Full house': 6,
                   'Four of a kind': 10, 'Straight flush': 15, 'Royal flush': 25}
    }

    strength = evaluate_hand(hand)

    if position == 'top':
        hand_ranks = ''.join(sorted([card.rank for card in hand], key=lambda x: rank_values[x]))
        return top_royalties.get(hand_ranks, 0)
    else:
        hands_map = {8: 'Straight flush', 7: 'Four of a kind', 6: 'Full house',
                     5: 'Flush', 4: 'Straight', 3: 'Three of a kind'}
        return middle_bottom_royalties[position].get(hands_map.get(strength[0]), 0)


class Deck:
    ranks = '23456789TJQKA'
    suits = 'CDHS'

    def __init__(self):
        self.cards = [Card(r, s) for r in self.ranks for s in self.suits]
        random.shuffle(self.cards)

    def draw(self, n=1):
        return [self.cards.pop() for _ in range(n)]


class OFCHand:
    def __init__(self):
        self.top = []
        self.middle = []
        self.bottom = []

    def place_cards(self, cards, positions):
        # positions = [('top', card), ('middle', card), ('bottom', card)]
        for pos, card in positions:
            getattr(self, pos).append(card)

    def valid_hand(self):
        if not self.is_complete():
            return False
        top_strength = evaluate_hand(self.top)
        middle_strength = evaluate_hand(self.middle)
        bottom_strength = evaluate_hand(self.bottom)
        return bottom_strength >= middle_strength >= top_strength

    def is_complete(self):
        return len(self.top) == 3 and len(self.middle) == 5 and len(self.bottom) == 5
    
    def count_points(self):
        top_strength = evaluate_hand(self.top)
        middle_strength = evaluate_hand(self.middle)
        bottom_strength = evaluate_hand(self.bottom)
        return (bottom_strength, middle_strength, top_strength)

class OpenFaceChinesePoker:
    def __init__(self):
        self.deck = Deck()
        self.player_hand = OFCHand()
        self.bot_hand = OFCHand()

    def initial_deal(self):
        return self.deck.draw(5), self.deck.draw(5)

    def play_round(self, player_moves, bot_moves):
        self.player_hand.place_cards(player_moves['cards'], player_moves['positions'])
        self.bot_hand.place_cards(bot_moves['cards'], bot_moves['positions'])

    def deal_next_cards(self):
        return self.deck.draw(1), self.deck.draw(1)

    def game_over(self):
        return self.player_hand.is_complete() and self.bot_hand.is_complete()
    
    def calculate_scores(self):
        player_points, bot_points = 0, 0

        # Determine fouls
        fouls = [not self.player_hand.valid_hand(), not self.bot_hand.valid_hand()]

        if not fouls[0] and not fouls[1]:
            player_points += 6
            bot_points += 6
        elif not fouls[0]:
            player_points += 6
            bot_points -= 6
        elif not fouls[1]:
            player_points -= 6
            bot_points += 6

        player_hand_values = self.player_hand.count_points() if not fouls[0] else ((0,[0]),)*3
        bot_hand_values = self.bot_hand.count_points() if not fouls[1] else ((0,[0]),)*3

        if not fouls[0]: 
            player_points += calculate_royalties(self.player_hand.top, "top")
            player_points += calculate_royalties(self.player_hand.middle, "middle")
            player_points += calculate_royalties(self.player_hand.bottom, "bottom")
        if not fouls[1]:
            bot_points += calculate_royalties(self.bot_hand.top, "top")
            bot_points += calculate_royalties(self.bot_hand.middle, "middle")
            bot_points += calculate_royalties(self.bot_hand.bottom, "bottom")

        player_hands_won = 0
        bot_hands_won = 0

        for i in range(3):
            winner = compare_hands(player_hand_values[i], bot_hand_values[i])
            if winner == 1:
                player_points += 1
                player_hands_won += 1
            elif winner == 2:
                bot_points += 1
                bot_hands_won += 1

        if player_hands_won == 3:
            player_points += 3
        if bot_hands_won == 3:
            bot_points += 3

        return player_points, bot_points

    def simulate_game(self, player_agent, bot_agent, initial_deal=True):
        if initial_deal:
            player_initial, bot_initial = self.initial_deal()
            print("Player initial cards:", player_initial)
            print("Bot initial cards:", bot_initial)

            # Place the first five cards. Each player receives (their hand, drawn card, opponent hand)
            # Hence, open face chinese poker
            player_move = player_agent(self.player_hand, player_initial, self.bot_hand)
            bot_move = bot_agent(self.bot_hand, bot_initial, self.player_hand)
            self.play_round(player_move, bot_move)

        while not self.game_over():
            player_card, bot_card = self.deal_next_cards()

            # Player would choose move manually or via agent
            player_move = player_agent(self.player_hand, player_card, self.bot_hand)

            # Bot move via agent provided
            bot_move = bot_agent(self.bot_hand, bot_card, self.player_hand)

            self.play_round(player_move, bot_move)

        print("Game over")
        print("Player Final hand:", self.player_hand.__dict__)
        print("Bot Final hand:", self.bot_hand.__dict__)
        print("")

        return self.calculate_scores()
    
# Simple agent example for bot decisions
def random_bot_agent(hand, card, opponent_hand):
    positions = ['top', 'middle', 'bottom']
    available_positions = [
        p for p in positions if len(getattr(hand, p)) < (3 if p == 'top' else 5)
    ]

    if not available_positions:
        return {'cards': [], 'positions': []}  # Return empty move if no spots left

    chosen_pos = random.choice(available_positions)
    return {'cards': [card], 'positions': [(chosen_pos, card)]}