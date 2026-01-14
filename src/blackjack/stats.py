from dataclasses import dataclass
from fractions import Fraction
from enum import StrEnum
from .constants import *

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
    payout: Fraction

class Stats:
    def __init__(self, rules:Rules, bankroll: int):
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