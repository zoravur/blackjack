import sys
import io
import pygame
import random
import math

from .deck import Deck
from .constants import *
from .util import *
from .hand import Hand
from .stats import Rules, Stats, HandResult, Outcome
from .policy import Action, dealerPolicyH17, dealerPolicyS17

class CardStack:
    def __init__(self):
        self.reset([])

    def reset(self, cards: list):
        self.cards = cards
        self.faceup = False
        self.cutCard = None
        self.thickness = 0.5
        self.cutCardSeen = False

    def flip(self):
        self.faceup = not self.faceup
        self.cards = self.cards[::-1]
    
    def draw(self):
        if len(self.cards)-1 == self.cutCard:
            self.cutCardSeen = True
            self.render()
            pygame.time.delay(1000)
            self.cards.pop()

        return self.cards.pop()

    def placeBottom(self, card):
        self.cards = [card] + self.cards

    def placeRandom(self, card):
        idx = random.randint(0, len(self.cards))
        self.cards = self.cards[:idx] + [card] + self.cards[idx:]

    def placeCutCard(self, card, startFrac: int = 0.2, endFrac: int = 0.4):
        N = len(self.cards)
        lo, hi = math.floor(N*startFrac), math.floor(N*endFrac)
        idx = random.randint(lo, hi)
        self.cutCardSeen = False
        self.cutCard = idx
        self.cards = self.cards[:idx] + [card] + self.cards[idx:]

    def placeTop(self, card):
        self.cards.append(card)
    
    def top(self):
        return self.cards[len(self.cards)-1]

    def render(self):
        h = len(self.cards)
        img = pygame.Surface((130, CARD_HEIGHT+h))
        img.fill(GREEN)

        if h == 0:
            return img, None

        for i, c in enumerate(self.cards):
            single = c.front if (self.faceup or (i == self.cutCard and i == len(self.cards)-1)) else c.back
            rect = single.get_rect(left=0, top=self.thickness*(h-i))
            img.blit(single, rect)
        return img, (0, self.thickness*i)
    
    def shuffle(self):
        for i in range(len(self.cards)-1):
            j = random.randint(i, len(self.cards)-1)
            self.cards[i], self.cards[j] = self.cards[j], self.cards[i]

class Table:
    def __init__(self, screen):
        self.screen = screen
        self.stacks = []
        self.positions = []

    def placeStack(self, stack, position):
        self.stacks.append(stack) 
        self.positions.append(position)

    def render(self):
        for stack, pos in zip(self.stacks, self.positions):
            img, anchor = stack.render()
            if anchor is None:
                anchor = (0, 0)
            rect = img.get_rect(midbottom=(pos[0], pos[1]+CARD_HEIGHT/2+anchor[1]))
            self.screen.blit(img, rect)

class Blackjack:
    def __init__(self, bankroll: int, rules: Rules, screen: pygame.Surface, *, dealerPolicy, playerPolicy):
        self.rules = rules
        self.screen = screen

        self.dealerHand = Hand()
        self.playerHand = Hand()

        self.dealerPolicy = dealerPolicy
        self.playerPolicy = playerPolicy

        self.shoe = CardStack()

        self.table = Table(self.screen)

        self.table.placeStack(self.dealerHand, position=(400,150))
        self.table.placeStack(self.playerHand, position=(400,650))
        self.table.placeStack(self.shoe, position=(150,400))

        self.stats = Stats(self.rules, bankroll)
        self.outcome = None
        self.clock = pygame.time.Clock()

    def turn(self):
        pass

    def step(self):
        pass

    def setupShoe(self):
        self.shoe.reset(Deck.fullDeckCards() * 6)
        self.shoe.shuffle()
        self.shoe.placeCutCard(self.shoe.draw())

    def render(self, delay=500):
        self.screen.fill(GREEN)
        self.table.render()

        for text, color, kwargs in self.stats.hud_text():
            self.screen.blit(*render_text(text, color, **kwargs))

        if self.outcome:
            match self.outcome:
                case Outcome.BUST:
                    color = RED
                case Outcome.LOSE:
                    color = RED
                case Outcome.PUSH:
                    color = (255, 255, 255)
                case Outcome.WIN:
                    color = (0, 255, 0)
                case Outcome.BLACKJACK:
                    color = (0, 0, 0)

            self.screen.blit(*render_text(self.outcome, color, center=(400,400)))
        self.screen.blit(*render_text(self.dealerHand.valueStr(), BLACK, center=(400, 300)))
        self.screen.blit(*render_text(self.playerHand.valueStr(), BLACK, center=(400, 500)))

        pygame.display.flip()

        pygame.time.delay(delay)


    def deal(self):
        self.dealerHand.clear()
        self.playerHand.clear()

        self.dealerHand.addCard(self.shoe.draw())
        self.playerHand.addCard(self.shoe.draw())
        self.dealerHand.addCard(self.shoe.draw())
        self.playerHand.addCard(self.shoe.draw())

    
    def getBet(self):
        return self.stats.bet(100) # TODO control bet sizing

    def hand(self) -> HandResult:
        # setup hand
        self.outcome = None
        bet = self.getBet()
        self.deal()

        self.render()

        if self.playerHand.blackjack():
            if self.dealerHand.blackjack():
                return HandResult(Outcome.PUSH, bet)
            else:
                return HandResult(Outcome.BLACKJACK, bet)
            
        if self.dealerHand.blackjack():
            return HandResult(Outcome.LOSE, bet)
            
        while True:
            action = dealerPolicyS17(self.playerHand)
            if action == Action.STAND:
                break
            
            elif action == Action.HIT:
                self.playerHand.addCard(self.shoe.draw())
            else: 
                print("invalid action: should never see this")
            if self.playerHand.value() > 21:
                return HandResult(Outcome.BUST, bet)
                
            self.render()
    
        while True:
            action = dealerPolicyH17(self.dealerHand)
            if action == Action.STAND:
                break
            elif action == Action.HIT:
                self.dealerHand.addCard(self.shoe.draw())
            else: 
                print("invalid action: should never see this")
            if self.dealerHand.value() > 21:
                return HandResult(Outcome.WIN, bet)
            self.render()

        if not self.outcome:
            if self.dealerHand.value() == self.playerHand.value():
                return HandResult(Outcome.PUSH, bet)
            elif self.dealerHand.value() > self.playerHand.value():
                return HandResult(Outcome.LOSE, bet)
            elif self.dealerHand.value() < self.playerHand.value():
                return HandResult(Outcome.WIN, bet)
            else:
                raise Exception("Invalid outcome")

    def play(self):
        self.setupShoe()
        self.render()

        while True:
            (self.handle(event) for event in pygame.event.get())
            handResult = self.hand()
            self.outcome = handResult.outcome
            self.stats.turn(handResult)
            self.render(1000)
            if self.shoe.cutCardSeen:
                self.setupShoe()

    def handle(self, event: pygame.event):
        if event.type == pygame.QUIT: sys.exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_a:
                self.playerHand.addCard(self.shoe.draw())