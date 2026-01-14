from enum import StrEnum
from .hand import Hand

class Action(StrEnum):
    HIT='HIT'
    STAND='STAND'
    # TODO: double down, split, surrender, etc.

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