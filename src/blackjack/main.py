import pygame
from fractions import Fraction
from .game import Blackjack
from .stats import Rules
from .gui import GameConfig, BlackjackGUI
from .policy import dealerPolicyH17, dealerPolicyS17

from .constants import *

def main():
    gc = GameConfig(gui=True, 
                    bankroll=5000,
                    rules=Rules(Fraction(3/2)),
                    dealerPolicy=dealerPolicyH17, 
                    playerPolicy=dealerPolicyS17)

    game = BlackjackGUI(gc)

    game.play()

if __name__ == "__main__":
    main()