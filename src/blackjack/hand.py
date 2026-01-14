import pygame
from .constants import *

class Hand:
    hand_offset = 20

    def __init__(self):
        self.cards = []

    def addCard(self, card):
        self.cards.append(card)

    def clear(self):
        self.cards = []

    def render(self):
        w = len(self.cards)
        img = pygame.Surface((self.hand_offset*w+CARD_WIDTH, CARD_HEIGHT))
        img.fill(GREEN)
        for i, c in enumerate(self.cards):
            single = c.front
            img.blit(single, (i*self.hand_offset, 0))
        return img, None

    def has_ace(self) -> int:
        return any(card.is_ace for card in self.cards)
    
    def total(self) -> int:
        return sum(card.value for card in self.cards)

    def valueStr(self):
        return ('S' if self.soft() else 'H') + str(self.value())
        
    def soft(self):
        return self.has_ace() and self.total() <= 11

    def value(self):
        if self.soft():
            return self.total() + 10
        else:
            return self.total()
    
    def blackjack(self):
        return self.value() == 21 and len(self.cards) == 2