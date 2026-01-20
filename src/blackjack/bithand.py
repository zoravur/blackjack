import numpy as np

MASK_TOTAL = np.uint16(0x00FF)
MASK_ACES  = np.uint16(0x0F00)  # 4 bits for aces
MASK_UPCARD = np.uint16(0xF000)

TOTAL_BITS = 8
ACE_SHIFT = 8

ACE_BITS = 4
UPCARD_SHIFT = 12
UPCARD_BITS = 4

def make_card(rank: np.ndarray) -> np.ndarray:
    ((rank << UPCARD_SHIFT) & MASK_UPCARD) | (
        ((rank == 1) << ACE_SHIFT) & MASK_ACES) + (rank & MASK_TOTAL)

def add_card(hand: np.ndarray | np.ndarray, card: np.ndarray) -> None:
    # mask = MASK_ACES | MASK_TOTAL
    # update = hand | (card & mask)
    # return np.where(hand == 0, card, update)
    mask0 = (hand == 0)                 # boolean temp
    hand |= (card & (MASK_ACES | MASK_TOTAL))  # in-place masked merge
    hand[mask0] = card[mask0]           # overwrite the “first card” cases

def hard_total(h: np.ndarray) -> np.array:
    return h & MASK_TOTAL

def ace_count(h: np.ndarray) -> np.array:
    return (h & MASK_ACES) >> ACE_SHIFT

def best_total(h: np.ndarray) -> np.array:
    t = hard_total(h)
    a = ace_count(h)
    if a > 0 and t + 10 <= 21:
        return t + 10
    return t

def soft(h: np.ndarray) -> np.array:
    t = hard_total(h)
    a = ace_count(h)
    return a > 0 and t + 10 <= 21

def upcard(h: np.ndarray) -> np.array:
    return (h | MASK_UPCARD) >> UPCARD_SHIFT