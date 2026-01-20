from .policy import Agent
import numpy as np
from dataclasses import dataclass
from . import bithand as hand


# --- numpy-native bitflags (uint8 everywhere) --------------------------------
class F:
    DONE      = np.uint8(1 << 0)
    TRUNCATED = np.uint8(1 << 1)
    P_TURN    = np.uint8(1 << 2)
    D_TURN    = np.uint8(1 << 3)
    P_STAND   = np.uint8(1 << 4)
    D_STAND   = np.uint8(1 << 5)

    # useful masks
    TURN_MASK  = np.uint8(P_TURN | D_TURN)
    STAND_MASK = np.uint8(P_STAND | D_STAND)


@dataclass
class EnvState:
    B: int
    n_decks: int
    total: int

    decks: np.ndarray      # (B, 10) float-ish counts
    ph: np.ndarray         # (B,) uint16 encoded hands
    dh: np.ndarray         # (B,) uint16 encoded hands
    flags: np.ndarray      # (B,) uint8 bitflags
    phase: np.ndarray      # (B,) uint32
    obs_mask: np.ndarray   # (B,) bool

    def __init__(self, B, n_decks):
        self.B = int(B)
        self.n_decks = int(n_decks)

        # rank counts: 1..9 have 4 per deck, 10 has 16 per deck
        self.decks = np.ones((self.B, 10), dtype=np.int16) * 4
        self.decks[:, 9] = 16
        self.decks *= self.n_decks

        # total cards remaining (lockstep scalar)
        self.total = int(self.decks.sum(axis=-1)[0])
        assert self.total == 52 * self.n_decks

        self.ph = np.zeros((self.B,), dtype=np.uint16)
        self.dh = np.zeros((self.B,), dtype=np.uint16)
        self.flags = np.zeros((self.B,), dtype=np.uint8)
        self.phase = np.zeros((self.B,), dtype=np.uint32)
        self.obs_mask = np.zeros((self.B,), dtype=np.bool_)

    def obs(self) -> tuple[np.ndarray, np.ndarray]:
        obs = np.stack(
            [
                hand.upcard(self.dh),
                hand.best_total(self.ph),
                hand.soft(self.ph),
            ],
            axis=-1,
        )
        return obs, self.obs_mask

    def draw(self, rng) -> np.ndarray:
        """
        Draw one card for every element in batch, updating deck counts.
        Returns encoded cards (same shape as (B,)).
        """
        # choose uniform integer in [0, total)
        u = np.floor(rng.random(self.B) * self.total).astype(np.int32)

        cs = self.decks.cumsum(axis=-1)  # (B, 10)
        k = (cs > u[:, None]).argmax(axis=-1).astype(np.int32)  # (B,)

        self.decks[np.arange(self.B), k] -= 1
        self.total -= 1

        return hand.make_card(k + 1)


class BatchedEnv:
    def reset(self, seed=0, batch_size: int = 2**16, n_decks: int = 8):
        self.rng = np.random.default_rng(seed)
        self.state = EnvState(batch_size, n_decks)

    def step(self, action, action_mask) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        action: (B,) int {0=hit, 1=stand} (or any convention you use)
        action_mask: (B,) bool, must match current obs_mask
        """
        s = self.state
        assert np.all(s.obs_mask == action_mask)

        # normalize input dtypes
        action = np.asarray(action)
        action_mask = np.asarray(action_mask, dtype=np.bool_)

        next_card = s.draw(self.rng)

        flags = s.flags          # uint8 view
        phase = s.phase
        ph = s.ph
        dh = s.dh
        B = s.B

        # --- dealing phase ----------------------------------------------------
        dealing = phase < 4
        deal_dealer = (phase % 2 == 1) & dealing
        deal_player = (phase % 2 == 0) & dealing

        # --- update flags (numpy uint8 only) ---------------------------------
        # player stands if they choose action==1 when a decision is required
        stand_now = (action == 1) & action_mask
        flags[stand_now] |= F.P_STAND

        # dealer stands on 17+
        dealer_stand = hand.best_total(dh) >= 17
        flags[dealer_stand] |= F.D_STAND

        # after initial deal, player turn begins
        flags[phase == 4] |= F.P_TURN

        # player hits if:
        # - not stood
        # - action_mask says env wants action
        # - it's player's turn
        # - not still dealing
        p_hit = (
            ((flags & F.P_STAND) == 0)
            & action_mask
            & ((flags & F.P_TURN) != 0)
            & ~dealing
        )

        # if player does NOT hit, end player turn
        flags[~p_hit] &= np.uint8(~F.P_TURN)

        # once player turn is over (phase>4), dealer turn begins
        start_dealer_turn = ((flags & F.P_TURN) == 0) & (phase > 4)
        flags[start_dealer_turn] |= F.D_TURN

        # dealer hits if:
        # - not stood
        # - it's dealer's turn
        # - not dealing
        d_hit = (
            ((flags & F.D_STAND) == 0)
            & ((flags & F.D_TURN) != 0)
            & ~dealing
        )

        # --- apply draws ------------------------------------------------------
        p_mask = deal_player | p_hit
        d_mask = deal_dealer | d_hit

        ph[p_mask] = hand.add_card(ph[p_mask], next_card[p_mask])
        dh[d_mask] = hand.add_card(dh[d_mask], next_card[d_mask])
        phase[p_mask | d_mask] += 1

        # --- terminal logic ---------------------------------------------------
        stand = ((flags & F.P_STAND) != 0) & ((flags & F.D_STAND) != 0)

        p_score = hand.best_total(ph)
        d_score = hand.best_total(dh)

        # after 2 cards each (phase==4): check blackjack state
        player_blackjack = (p_score == 21) & (phase == 4)
        dealer_blackjack = (d_score == 21) & (phase == 4)

        blackjack_win = player_blackjack & ~dealer_blackjack
        lose = (p_score > 21) | (stand & (d_score > p_score)) | dealer_blackjack
        push = (p_score == d_score) & (stand | player_blackjack | dealer_blackjack)
        win = (p_score <= 21) & ((d_score > 21) | (stand & (d_score < p_score)))

        round_finished_mask = win | lose | push | blackjack_win

        rewards = np.zeros(B, dtype=np.float32)
        rewards[blackjack_win] = 1.5
        rewards[push] = 0.0
        rewards[win] = 1.0
        rewards[lose] = -1.0

        # --- reset finished rounds -------------------------------------------
        phase[round_finished_mask] = 0
        ph[round_finished_mask] = 0
        dh[round_finished_mask] = 0
        flags[round_finished_mask] = 0

        # --- immediately begin dealing new rounds for finished lanes ----------
        start_new = round_finished_mask

        dealing2 = start_new & (phase < 4)
        deal_dealer2 = (phase % 2 == 1) & dealing2
        deal_player2 = (phase % 2 == 0) & dealing2

        # reuse next_card: "don't worry about other bugs right now"
        ph[deal_player2] = hand.add_card(ph[deal_player2], next_card[deal_player2])
        dh[deal_dealer2] = hand.add_card(dh[deal_dealer2], next_card[deal_dealer2])
        phase[deal_player2 | deal_dealer2] += 1

        # env requests action only on player turn
        s.obs_mask = (flags & F.P_TURN) != 0

        return s.obs(), rewards
