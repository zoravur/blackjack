from .policy import Agent
import numpy as np
from enum import Enum
from dataclasses import dataclass
from . import bithand as hand

class Flags(Enum):
    DONE = 1 << 0
    TRUNCATED = 1 << 1
    P_TURN = 1 << 2
    D_TURN = 1 << 3
    P_STAND = 1 << 4
    D_STAND = 1 << 5

@dataclass
class EnvState:
    B: int # batch dim
    n_decks: int
    total: int # number of cards left

    decks: np.ndarray # np.array[10] -- rank counts
    ph: np.ndarray # int -- player hands
    dh: np.ndarray # int -- dealer hands
    flags: np.ndarray # int -- simulation control flags
    phase: np.ndarray # int -- how many cards we are into the shoe
    obs_mask: np.ndarray # bool -- the runs which require an action

    def __init__(self, B, n_decks):
        self.B = B
        self.n_decks = n_decks

        self.decks = np.ones((self.B, 10)) * self.n_decks 
        self.decks[:, 9] = 16
        self.decks *= self.n_decks
        self.total = self.decks.sum(axis=-1)[0]
        assert np.all(self.total == 52*self.n_decks)

        self.ph = np.zeros((self.B), dtype=np.uint16)
        self.dh = np.zeros((self.B), dtype=np.uint16)
        self.flags = np.zeros((self.B), dtype=np.uint8)
        self.phase = np.zeros((self.B), dtype=np.uint32)
        self.obs_mask = np.zeros((self.B), dtype=np.bool)

    def obs(self) -> tuple[np.ndarray, np.ndarray]:
        return np.stack([hand.upcard(self.dh), 
                         hand.best_total(self.ph), 
                         hand.soft(self.ph)], axis=-1), self.obs_mask

    def draw(self, rng):
        u = np.random.floor(rng.random(self.decks.shape[:-1]) * self.total)

        cs = self.decks.cumsum(axis=-1)
        k = (cs > u[:, None]).argmax(axis=-1) 

        self.decks[np.arange(self.decks.shape[0]), k] -= 1 # scatter decrement
        self.total -= 1 # total might be a scalar if we're doing true lockstep
        return hand.make_card(k+1)

class BatchedEnv():
    def reset(self, seed=0, batch_size: int=2**16, n_decks: int=8):
        self.rng = np.random.default_rng(seed)
        self.state = EnvState(batch_size, n_decks)

    def step(self, action, action_mask) -> EnvState:
        assert np.all(self.state.obs_mask == action_mask)

        next_card = self.state.draw(self.rng)
        flags = self.state.flags
        phase = self.state.phase
        ph = self.state.ph
        dh = self.state.dh
        B = self.state.B

        dealing = phase < 4
        deal_dealer = phase % 2 == 1 and dealing
        deal_player = phase % 2 == 0 and dealing

        flags |= Flags.P_STAND and action == 1 and action_mask
        flags |= Flags.D_STAND and hand.best_total(dh) < 17
        flags |= Flags.P_TURN and phase == 4

        p_hit = (flags & Flags.P_STAND) == 0 and action_mask & flags & Flags.P_TURN and not dealing

        flags &= ~Flags.P_TURN or p_hit
        flags |= Flags.D_TURN and (flags & Flags.P_TURN) == 0 and phase > 4

        d_hit = (flags & Flags.D_STAND) == 0 and flags & Flags.D_TURN and not dealing

        p_mask = deal_player or p_hit
        d_mask = deal_dealer or d_hit

        hand.add_card(ph[p_mask], next_card[p_mask])
        hand.add_card(dh[d_mask], next_card[d_mask])

        # all cards dealt
        stand = (flags & (Flags.P_STAND | Flags.D_STAND)) == 0 

        p_score = hand.best_total(ph)
        d_score = hand.best_total(dh)

        player_blackjack = p_score == 21 and phase == 3 # phase == 3 -> check after deal
        dealer_blackjack = d_score == 21 and phase == 3 # phase == 3 -> check after deal

        blackjack_win = player_blackjack
        lose = p_score > 21 or (stand and d_score > p_score) or dealer_blackjack
        push = (p_score == d_score) and (stand or player_blackjack)
        win = d_score > 21

        round_finished_mask = win or lose or push or player_blackjack

        rewards = np.zeros(B)
        rewards[blackjack_win] = 1.5
        rewards[lose] = -1.0
        rewards[push] = 0.0
        rewards[win] = 1.0

        phase[round_finished_mask] = 0
        ph[round_finished_mask] = 0
        dh[round_finished_mask] = 0 
        flags[round_finished_mask] = 0

        hand.add_card(ph[stand], next_card[stand])
        phase[stand] += 1
        
        self.state.obs_mask = flags & Flags.P_TURN
        return self.state.obs(), rewards