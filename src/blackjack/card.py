from .util import load_svg
from dataclasses import dataclass

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