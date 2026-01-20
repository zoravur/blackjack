from enum import StrEnum, Enum
from .hand import Hand
import numpy as np
import math

ERR_EPS = 0.0000001

class Action(Enum):
    HIT = np.uint8(0)
    STAND = np.uint8(1)
    # TODO: double down, split, surrender, etc.

def init_value_table():
    return np.zeros((10, 10, 2, 2), dtype=np.float64)

# def value_to_fixed_policy(value_table: ValueTable):
#     return np.argmax(value_table, axis=-1)

def split_observation_matrix(observations: np.ndarray):
    i = observations[..., 0]
    j = observations[..., 1]
    s = observations[..., 2]
    return i, j, s

def sample_epsilon_greedy(rng: np.random.Generator, value_table: np.ndarray, eps: np.float64, observations: np.ndarray):
    i, j, s = split_observation_matrix(observations)

    # a = np.zeros_like(observations.shape[:-1], dtype=np.uint8)
    # a[j < 12] = 0

    qs = value_table[i, j, s, :]  # shape (..., 2)
    batch_shape = qs.shape[:-1]

    explore = rng.random(batch_shape) < eps          # per-state epsilon coin flip
    rand_a   = rng.integers(0, 2, size=batch_shape)  # random actions

    # greedy with random tie-break
    greedy_a = np.argmax(qs + 1e-12 * rng.random(qs.shape), axis=-1)

    a = np.where(explore, rand_a, greedy_a).astype(np.uint8)

    
    assert np.all(a <= 1)
    
    return a

# def calculate_total_reward(rewards: np.ndarray):
#     # gamma = 1 (assumed)
#     return np.flip(np.cumsum(np.flip(rewards, -1), -1), -1)

# def calculate_total_reward(rewards: np.ndarray, gamma: float=0.95):
#     # Initialize an array of zeros with float type
#     returns = np.zeros_like(rewards, dtype=np.float64)
#     running_reward = 0
    
#     # Iterate backwards through the rewards
#     for t in reversed(range(len(rewards))):
#         running_reward = rewards[t] + gamma * running_reward
#         returns[t] = running_reward
        
#     return returns

# import numpy as np

def calculate_total_reward(rewards: np.ndarray, gamma: float = 0.9):
    # Ensure shape is (Batch, Time)
    if rewards.ndim == 1:
        rewards = rewards[np.newaxis, :]  # Add batch dim if missing
        
    B, T = rewards.shape
    returns = np.zeros_like(rewards, dtype=np.float64)
    
    # We need a running total for EACH item in the batch
    running_reward = np.zeros(B, dtype=np.float64)
    
    # Loop backwards through TIME (T), not Batch
    for t in reversed(range(T)):
        # Update the running total for the whole batch at this timestep
        running_reward = rewards[:, t] + gamma * running_reward
        returns[:, t] = running_reward
        
    return returns

class Agent:
    def __init__(self):
        # We no longer need alpha for the update, as it is determined by counts
        self.q = init_value_table()
        # Initialize counts to 0. Using float to avoid integer division gotchas later
        self.counts = np.zeros_like(self.q, dtype=np.float64) 
        self.k = 0

    def eps(self):
        # Your existing epsilon decay
        return 1.0 / math.sqrt(self.k if self.k > 0 else 1)
    
    def sample(self, rng: np.random.Generator, observations, training: bool):
        self.k += 1
        return sample_epsilon_greedy(rng, self.q, self.eps() if training else 0, observations)
        
    def trajectory_update(
        self,
        actions: np.ndarray,         # (B,T)
        observations: np.ndarray,    # (B,T,3)
        rewards: np.ndarray,         # (B,T)
        mask: np.ndarray,            # (B,T) boolean, True = valid
    ):
        assert actions.shape == observations.shape[:-1] == rewards.shape == mask.shape, \
            "Trajectory shapes must match"

        # 1. Calculate Returns (Monte Carlo)
        G_bt = calculate_total_reward(rewards)  # (B,T)

        # 2. Flatten everything to process the whole batch as a list of samples
        i = observations[..., 0].reshape(-1)
        j = observations[..., 1].reshape(-1)
        s = observations[..., 2].reshape(-1)
        a = actions.reshape(-1)
        G = G_bt.reshape(-1)
        m = mask.reshape(-1).astype(bool)

        # 3. Filter out invalid steps (padding)
        i = i[m]
        j = j[m]
        s = s[m]
        a = a[m]
        G = G[m]

        if G.size == 0:
            return G_bt[:, 0]

        # 4. Create unique hash/index for every state-action pair
        I, J, S, A = self.q.shape
        flat = (((i.astype(np.uint64) * J + j.astype(np.uint64)) * S + s.astype(np.uint64)) * A + a.astype(np.uint64))

        # 5. Group by State-Action pair
        # uniq: unique state-action indices
        # inv: inverse indices to reconstruct original array
        # k_batch: how many times this (s,a) appeared in THIS batch
        uniq, inv, k_batch = np.unique(flat, return_inverse=True, return_counts=True)

        # 6. Calculate mean return for this batch for each unique pair
        Gsum = np.zeros(len(uniq), dtype=np.float64)
        np.add.at(Gsum, inv, G)
        Gmean = Gsum / k_batch

        # 7. Decode unique indices back to (i, j, s, a)
        a_u = uniq % A
        tmp = uniq // A
        s_u = tmp % S
        tmp //= S
        j_u = tmp % J
        i_u = tmp // J

        # 8. PERFORM THE STATIONARY UPDATE
        
        # Get historical counts and Q-values
        n_old = self.counts[i_u, j_u, s_u, a_u]
        q_old = self.q[i_u, j_u, s_u, a_u]
        
        # Formula: Q_new = (N_old * Q_old + sum(G_batch)) / (N_old + k_batch)
        # Note: sum(G_batch) is equivalent to k_batch * Gmean
        
        # To strictly avoid overflow if counts get massive, we can use the alpha form:
        # Effective alpha for this update = new_samples / total_samples
        total_n = n_old + k_batch
        
        # Update Q-Values
        # This is mathematically equivalent to averaging all returns ever seen
        self.q[i_u, j_u, s_u, a_u] = q_old + (k_batch / total_n) * (Gmean - q_old)

        # Update Counts
        self.counts[i_u, j_u, s_u, a_u] = total_n

        return G_bt[:, 0]
# class Agent:
#     def __init__(self, alpha):
#         self.alpha = alpha
#         self.q = init_value_table()
#         self.k = 0

#     def eps(self):
#         return 1 / math.sqrt(self.k)
    
#     def sample(self, rng: np.random.Generator, observations, training: bool):
#         self.k += 1
#         return sample_epsilon_greedy(rng, self.q, self.eps() if training else 0, observations)
        
#     def trajectory_update(
#         self,
#         actions: np.ndarray,        # (B,T)
#         observations: np.ndarray,    # (B,T,3)
#         rewards: np.ndarray,         # (B,T)
#         mask: np.ndarray,            # (B,T) boolean, True = valid
#     ):
#         assert actions.shape == observations.shape[:-1] == rewards.shape == mask.shape, \
#             "Trajectory shapes must match"

#         G_bt = calculate_total_reward(rewards)  # (B,T)

#         # Flatten everything
#         i = observations[..., 0].reshape(-1)
#         j = observations[..., 1].reshape(-1)
#         s = observations[..., 2].reshape(-1)
#         a = actions.reshape(-1)
#         G = G_bt.reshape(-1)
#         m = mask.reshape(-1).astype(bool)

#         # Filter out invalid steps
#         i = i[m]
#         j = j[m]
#         s = s[m]
#         a = a[m]
#         G = G[m]
#         # print("max(i)", np.max(i)) # 9

#         # print(f"{m.sum()}/{m.size}")
#         if G.size == 0:
#             # print("trajectory_update: NOTHING TO UPDATE")
#             return G_bt[:, 0]  # nothing to update

#         I, J, S, A = self.q.shape
#         # print("J,S,A:", J, S, A, "J*S*A:", J*S*A)
#         # print("i max:", i.max(), "j max:", j.max(), "s max:", s.max(), "a max:", a.max())
#         flat = (((i.astype(np.uint64) * J + j.astype(np.uint64)) * S + s.astype(np.uint64)) * A + a.astype(np.uint64))
#         # print("flat max:", flat.max())

#         uniq, inv, k = np.unique(flat, return_inverse=True, return_counts=True)

#         Gsum = np.zeros(len(uniq), dtype=np.float64)
#         np.add.at(Gsum, inv, G)
#         Gmean = Gsum / k

#         a_u = uniq % A
#         tmp = uniq // A
#         s_u = tmp % S
#         tmp //= S
#         j_u = tmp % J
#         i_u = tmp // J

#         # self.counts[i_u, j_u, s_u, a_u] += k

#         alpha_eff = 1.0 - (1.0 - self.alpha) ** k
#         q_old = self.q[i_u, j_u, s_u, a_u]
#         self.q[i_u, j_u, s_u, a_u] = q_old + alpha_eff * (Gmean - q_old)

#         return G_bt[:, 0]

    # def trajectory_update(self, actions: np.ndarray, observations: np.ndarray, rewards: np.ndarray):
    #     assert actions.shape == observations.shape[:-1] == rewards.shape, "Trajectory shapes must match"

    #     G_bt = calculate_total_reward(rewards)  # (B,T)

    #     i = observations[..., 0].reshape(-1)
    #     j = observations[..., 1].reshape(-1)
    #     s = observations[..., 2].reshape(-1)
    #     a = actions.reshape(-1)
    #     G = G_bt.reshape(-1)

    #     I, J, S, A = self.q.shape
    #     flat = (((i * J + j) * S + s) * A + a)

    #     uniq, inv, k = np.unique(flat, return_inverse=True, return_counts=True)

    #     Gsum = np.zeros(len(uniq), dtype=np.float64)
    #     np.add.at(Gsum, inv, G)
    #     Gmean = Gsum / k

    #     a_u = uniq % A
    #     tmp = uniq // A
    #     s_u = tmp % S
    #     tmp //= S
    #     j_u = tmp % J
    #     i_u = tmp // J

    #     self.counts[i_u, j_u, s_u, a_u] += k

    #     alpha_eff = 1.0 - (1.0 - self.alpha) ** k
    #     q_old = self.q[i_u, j_u, s_u, a_u]
    #     self.q[i_u, j_u, s_u, a_u] = q_old + alpha_eff * (Gmean - q_old)

    #     return G_bt[:, 0]
