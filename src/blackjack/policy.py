from enum import StrEnum, Enum
from .hand import Hand
import numpy as np


type VALUE_FLOAT_TYPE = np.float64
type ValueTable = np.ndarray # shape: (10 (dealer card), 10 (player total), 2 (soft/hard), 2 (HIT/STAND)) -> value
type PolicyTable = np.ndarray # shape: (10 (dealer card), 10 (player total), 2 (soft/hard)) -> action (int)

ERR_EPS = 0.0000001

class Action(Enum):
    HIT = np.uint8(0)
    STAND = np.uint8(1)
    # TODO: double down, split, surrender, etc.

def init_value_table():
    return np.zeros((10, 10, 2, 2), dtype=VALUE_FLOAT_TYPE)

def value_to_fixed_policy(value_table: ValueTable):
    return np.argmax(value_table, axis=-1)

def split_observation_matrix(observations: np.ndarray):
    i = observations[..., 0]
    j = observations[..., 1]
    s = observations[..., 2]
    return i, j, s

def sample_epsilon_greedy(rng: np.random.Generator, value_table: ValueTable, eps: np.float64, observations: np.ndarray):
    i, j, s = split_observation_matrix(observations)

    qs = value_table[i, j, s]
    if rng.random() < eps:
        a = rng.random.choice([0, 1], size=observations.shape)
    else:
        a = rng.random.choice(np.flatnonzero(qs == qs.max()))
    
    return a

def calculate_total_reward(rewards: np.ndarray):
    # gamma = 1 (assumed)
    return np.flip(np.cumsum(np.flip(rewards, -1), -1), -1)

class Agent:
    def __init__(self, alpha=0.0001, eps=0.1):
        self.alpha = 0.0001
        self.q = init_value_table()
        self.k = 0

    def eps(self):
        return 1 / self.k
    
    def sample(self, rng: np.random.Generator, observations, training: bool):
        self.k += observations.size
        return sample_epsilon_greedy(rng, self.q, self.eps() if training else 0, observations)
        
    def trajectory_update(self, actions: np.ndarray, observations: np.ndarray, rewards: np.ndarray):
        assert actions.shape == observations.shape[:-1] == rewards.shape, "Trajectory shapes must match"

        G_bt = calculate_total_reward(rewards)  # (B,T)

        i = observations[..., 0].reshape(-1)
        j = observations[..., 1].reshape(-1)
        s = observations[..., 2].reshape(-1)
        a = actions.reshape(-1)
        G = G_bt.reshape(-1)

        I, J, S, A = self.q.shape
        flat = (((i * J + j) * S + s) * A + a)

        uniq, inv, k = np.unique(flat, return_inverse=True, return_counts=True)

        Gsum = np.zeros(len(uniq), dtype=np.float64)
        np.add.at(Gsum, inv, G)
        Gmean = Gsum / k

        a_u = uniq % A
        tmp = uniq // A
        s_u = tmp % S
        tmp //= S
        j_u = tmp % J
        i_u = tmp // J

        self.counts[i_u, j_u, s_u, a_u] += k

        alpha_eff = 1.0 - (1.0 - self.alpha) ** k
        q_old = self.q[i_u, j_u, s_u, a_u]
        self.q[i_u, j_u, s_u, a_u] = q_old + alpha_eff * (Gmean - q_old)

        return G_bt[:, 0]

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
