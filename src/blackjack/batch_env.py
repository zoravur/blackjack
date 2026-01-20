from .policy import Agent
import numpy as np
from dataclasses import dataclass
from . import bithand as hand


# --- numpy-native bitflags (uint8 everywhere) --------------------------------
class F:
    #DONE      = np.uint8(1 << 0)
    #TRUNCATED = np.uint8(1 << 1)
    #P_TURN    = np.uint8(1 << 2)
    #D_TURN    = np.uint8(1 << 3)
    P_STAND   = np.uint8(1 << 4)
    D_STAND   = np.uint8(1 << 5)

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
        self.total = self.decks.sum(axis=-1).astype(np.int32)  # (B,)
        assert np.all(self.total == 52 * self.n_decks)

        self.ph = np.zeros((self.B,), dtype=np.uint16)
        self.dh = np.zeros((self.B,), dtype=np.uint16)
        self.flags = np.zeros((self.B,), dtype=np.uint8)
        self.phase = np.zeros((self.B,), dtype=np.uint32)
        self.obs_mask = np.zeros((self.B,), dtype=np.bool_)

    def obs(self) -> tuple[np.ndarray, np.ndarray]:
        obs = np.stack(
            [
                hand.upcard(self.dh).astype(np.uint8)-1,
                hand.best_total(self.ph).astype(np.uint8)-12,
                hand.soft(self.ph).astype(np.uint8),
            ],
            axis=-1,
            dtype=np.uint8
        )
        return obs, self.obs_mask

    def draw(self, rng) -> np.ndarray:
        """
        Draw one card for every element in batch, updating deck counts.
        Returns encoded cards (same shape as (B,)).
        """

        u = (rng.random(self.B) * self.total).astype(np.int32)  # (B,)

        cs = self.decks.cumsum(axis=-1)
        k = (cs > u[:, None]).argmax(axis=-1).astype(np.int32)

        self.decks[np.arange(self.B), k] -= 1
        self.total -= 1

        row_totals = self.decks.sum(axis=-1)
        assert np.all(row_totals == self.total), (row_totals.min(), row_totals.max(), self.total)
        assert np.all(self.decks >= 0)

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
        # print(hand.hard_total(next_card.max()), hand.hard_total(next_card.min()))

        flags = s.flags          # uint8 view
        phase = s.phase
        ph = s.ph
        dh = s.dh
        B = s.B

        # --- dealing phase ----------------------------------------------------
        dealing = phase < 4
        deal_dealer = (phase % 2 == 1) & dealing
        deal_player = (phase % 2 == 0) & dealing


        # --- policy (player) decisions ----------------------------------------
        # player stands if they choose action==1 when a decision is required
        player_stand = (action == 1) & action_mask
        flags[player_stand] |= F.P_STAND

        # dealer stands on 17+
        dealer_stand = hand.best_total(dh) >= 17
        flags[dealer_stand] |= F.D_STAND

        # player turn if:
        # - not stood
        # - not still dealing
        player_active = ~dealing & ((flags & F.P_STAND) == 0)
        # - not stood
        # - not still dealing
        # - player not active
        dealer_active = ~dealing & ((flags & F.D_STAND) == 0) & ~player_active

        # --- apply draws ------------------------------------------------------
        p_mask = deal_player | player_active
        d_mask = deal_dealer | dealer_active

        # all stand <=> neither player nor dealer is active <=> neither player gets a card
        all_stand = ~p_mask & ~d_mask
        #assert np.all(k), f"Mismatch at index {np.flatnonzero(k)[:10]}"
        assert np.all((((flags & F.P_STAND) != 0) & ((flags & F.D_STAND) != 0)) == all_stand)

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
        blackjack_lose = dealer_blackjack & ~player_blackjack
        lose = (p_score > 21) | (stand & (d_score > p_score)) | blackjack_lose
        push = (p_score == d_score) & (stand | player_blackjack | dealer_blackjack)
        win = ~blackjack_win & (p_score <= 21) & ((d_score > 21) | (stand & (d_score < p_score)))

        round_finished_mask = win | lose | push | blackjack_win

        ############# assertions code ###############
        outcomes = np.stack([blackjack_win, push, win, lose], axis=0)   # (4,B)
        k = outcomes.sum(axis=0)                                        # (B,)

        # 0 allowed for nonterminal steps; terminal steps should have exactly 1
        assert np.all((k == 0) | (k == 1)), f"Overlapping outcomes in {np.flatnonzero(k > 1)[:10]}"
        #############################################

        rewards = np.zeros(B, dtype=np.float64)
        rewards[blackjack_win] = 1.5
        rewards[push] = 0.0
        rewards[win] = 1.0
        rewards[lose] = -1.0

        # --- reset finished rounds -------------------------------------------
        phase[round_finished_mask] = 0
        ph[round_finished_mask] = 0
        dh[round_finished_mask] = 0
        flags[round_finished_mask] = 0

        # --- immediately begin dealing unused cards ----------
        ph[all_stand] = hand.add_card(ph[all_stand], next_card[all_stand])
        phase[all_stand] += 1

        # print(p_mask.sum())
        # print(d_mask.sum())
        # print(all_stand.sum())
        # # these sum to 4096
        # print("\n")
        # below assert passes
        assert np.all((all_stand.astype(np.uint8) + p_mask.astype(np.uint8) + d_mask.astype(np.uint8)) == 1)

        # env requests action only on player turn
        needs_action = (phase >= 4) & ((flags & F.P_STAND) == 0) & (hand.best_total(ph) >= 12)
        s.obs_mask = needs_action

        return s.obs(), rewards
