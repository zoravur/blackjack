from .card import Card

RANKS = list("23456789JQKA") + ['10']
SUITS = list("HDSC")

class Deck:
    def __init__(self):
        self.cards = Deck.fullDeckCards()
        
    def shuffle(self):
        for i in range(len(self.cards)-1):
            j = random.randint(i, len(self.cards)-1)
            self.cards[i], self.cards[j] = self.cards[j], self.cards[i]

    def __getitem__(self, i):
        return self.cards[i]

    @staticmethod
    def fullDeckCards():
        return [Card(r, s) for r in RANKS for s in SUITS]