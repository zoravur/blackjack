from .policy import Agent
from .batch_env import BatchedEnv
import numpy as np

def main():
    env = BatchedEnv()
    agent = Agent()

    ss = np.random.SeedSequence(1337)
    seeds = ss.spawn(16)

    mean_rewards = []
    for i in range(16):
        env_seed, agent_seed = seeds[i].spawn(2)
        env.reset(seed=env_seed)

        rng = np.random.default_rng(agent_seed)
        obs, mask = env.state.obs()

        obss, actions, masks, rewards = [], [], [], []
        for _ in range(env.state.n_decks * 52):
            obss.append(obs)
            masks.append(mask)
            action = agent.sample(rng, obs, training=True)
            actions.append(action)
            (obs, mask), reward = env.step(action, mask)
            rewards.append(reward)
        obss = np.stack(obss, axis=1)
        actions = np.stack(actions, axis=1)
        masks = np.stack(masks, axis=1)
        rewards = np.stack(rewards, axis=1)
        G = agent.trajectory_update(actions, obss, rewards, masks)
        mean_rewards.append(G.mean())
    print(mean_rewards)

if __name__ == "__main__":
    main()