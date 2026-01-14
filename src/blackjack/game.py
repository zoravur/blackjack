import sys
import pygame
import io
import cairosvg
import random
import functools
import math
from fractions import Fraction
from enum import StrEnum, auto
from dataclasses import dataclass


CARD_HEIGHT = 182

@functools.cache
def load_svg(path, size=None):
    if size is None:
        w, h = None, None
    else:
        w, h = size
    png_bytes = cairosvg.svg2png(url=path, output_width=130, output_height=CARD_HEIGHT)
    return pygame.image.load(io.BytesIO(png_bytes)).convert_alpha()

@functools.cache
def get_font(size=48):
    return pygame.font.Font(None, size)

def render_text(s, color, **posn):
    font = get_font()
    surf = font.render(s, True, color)
    rect = surf.get_rect(**posn)
    return surf, rect

RANKS = list("23456789JQKA") + ['10']
SUITS = list("HDSC")
GREEN = (85, 170, 85)
GOLD = (255, 215, 0)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
WIDTH, HEIGHT = 800, 800

@dataclass
class Card:
    rank: str
    suit: str
    value: int = None
    is_ace: bool = False
    
    def __post_init__(self):
        self.front = load_svg(f"assets/full-deck/{self.rank}{self.suit}.svg")
        self.back = load_svg(f"assets/full-deck/BB.svg")

        if self.rank.isdigit():
            self.value = int(self.rank)
        elif self.rank in 'JQK':
            self.value = 10
        else:
            self.value = 1
            self.is_ace = True
        

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
        img = pygame.Surface((self.hand_offset*w+130, CARD_HEIGHT))
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

class Outcome(StrEnum):
    BUST='BUST'
    WIN='WIN'
    PUSH='PUSH'
    BLACKJACK='BLACKJACK'
    LOSE='LOSE'

@dataclass
class HandResult:
    outcome: Outcome
    bet: int

@dataclass
class Rules:
    payout: Fraction = Fraction(3,2)

class Action(StrEnum):
    HIT='HIT'
    STAND='STAND'
    # TODO: double down, split, surrender, etc.

class Stats:
    def __init__(self, rules:Rules=Rules(), bankroll: int=5000):
        self.rules = rules
        self.bankroll = bankroll
        self.roundCounts = dict(BUST=0, WIN=0, PUSH=0, BLACKJACK=0, LOSE=0)

    def calculatePayout(self, handResult: HandResult) -> int:
        outcome, bet = handResult.outcome, handResult.bet
        match outcome:
            case Outcome.BUST:
                return 0
            case Outcome.LOSE:
                return 0
            case Outcome.WIN:
                return bet + bet * self.rules.payout
            case Outcome.BLACKJACK:
                return bet + bet * self.rules.payout
            case Outcome.PUSH:
                return bet

    def turn(self, handResult: HandResult):
        self.roundCounts[handResult.outcome] += 1
        self.bankroll += self.calculatePayout(handResult)

    def bet(self, amount: int) -> int:
        amount = min(self.bankroll, amount)
        self.bankroll -= amount
        return amount

    def fmtBankroll(self) -> str:
        return f"${(self.bankroll / 100):.2f}"

    def hud_text(self):
        return [
            (f"Bank: {self.fmtBankroll()}", GOLD, dict(top=50, right=WIDTH-50)),
            (f"Round: {sum(self.roundCounts.values())}", BLACK, dict(top=100, right=WIDTH-50))
        ]
        
        
def dealerPolicyH17(hand: Hand) -> Action:
    if hand.value() >= 17:
        return Action.STAND
    else:
        return Action.HIT

def dealerPolicyS17(hand: Hand) -> Action:
    if (hand.value() >= 17 and not hand.soft()) or hand.value() >= 18: 
        return Action.STAND
    else:
        return Action.HIT

class Blackjack:
    def __init__(self, bankroll: int, rules: Rules, screen: pygame.Surface):
        self.rules = rules
        self.screen = screen

        self.dealerHand = Hand()
        self.playerHand = Hand()

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

def main():
    pygame.init()
    size = WIDTH, HEIGHT
    screen = pygame.display.set_mode(size)
    game = Blackjack(bankroll=5000, rules=Rules(Fraction(3, 2)), screen=screen)

    game.play()

if __name__ == "__main__":
    main()



        