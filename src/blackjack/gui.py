from dataclasses import dataclass
from fractions import Fraction
import pygame

from typing import Callable
from .constants import *
from .game import Blackjack
from .policy import Hand, Action
from .stats import Rules

@dataclass
class GameConfig:
    gui: bool
    bankroll: int
    rules: Rules
    dealerPolicy: Callable[[Hand], Action]
    playerPolicy: Callable[[Hand], Action]
    delayShort: int
    delayLong: int

class BlackjackGUI:
    def __init__(self, gc: GameConfig):
        self.screen = None
        self.gc = gc

        if self.gc.gui:
            self.startGui()
        self.game = Blackjack(bankroll=self.gc.bankroll, 
                              rules=self.gc.rules, 
                              screen=self.screen, 
                              dealerPolicy=self.gc.dealerPolicy, 
                              playerPolicy=self.gc.playerPolicy,
                              delayShort=self.gc.delayShort,
                              delayLong=self.gc.delayLong)

    def play(self):
        self.game.play()
    
    def startGui(self):
        pygame.init()
        size = WIDTH, HEIGHT
        self.screen = pygame.display.set_mode(size)