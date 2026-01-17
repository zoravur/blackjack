from .util import load_png
from dataclasses import dataclass

@dataclass
class Card:
    rank: str
    suit: str
    value: int = None
    is_ace: bool = False
    
    def __post_init__(self):
        self.front = load_png(f"assets/full-deck-png/{self.rank}{self.suit}.png")
        self.back = load_png(f"assets/full-deck-png/BB.png")

        if self.rank.isdigit():
            self.value = int(self.rank)
        elif self.rank in 'JQK':
            self.value = 10
        else:
            self.value = 1
            self.is_ace = True