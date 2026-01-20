from .policy import Agent
from .batch_env import BatchedEnv
import numpy as np
import matplotlib.pyplot as plt

def plot_category_pcts(history, categories=None, title="Category % over time"):
    # history: list[dict(category -> pct)]
    if categories is None:
        categories = sorted({k for d in history for k in d.keys()})

    steps = np.arange(len(history))
    Y = {c: np.array([d.get(c, 0.0) for d in history], dtype=float) for c in categories}

    plt.figure(figsize=(10, 5))
    for c in categories:
        plt.plot(steps, Y[c], label=str(c))

    plt.xlabel("Step")
    plt.ylabel("Percent")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    env = BatchedEnv()
    # agent = Agent(alpha=0.00001)
    agent = Agent()
    N_RUNS = 1024

    ss = np.random.SeedSequence(1400)
    seeds = ss.spawn(N_RUNS)

    mean_rewards = []
    policies = []
    history = []
    for i in range(N_RUNS):
        env_seed, agent_seed = seeds[i].spawn(2)
        env.reset(seed=env_seed, batch_size=2**1)

        rng = np.random.default_rng(agent_seed)
        obs, mask = env.state.obs()
        action = np.zeros(obs.shape[:-1], dtype=np.uint8)

        obss, actions, masks, rewards = [], [], [], []
        for j in range(env.state.n_decks * 52):
            obss.append(obs)
            masks.append(mask)
            action[mask] = agent.sample(rng, obs[mask], training=True)
            actions.append(action[:])
            (obs, mask), reward = env.step(action, mask)
            rewards.append(reward)

            # print(f"Step {j} complete")
    
        obss = np.stack(obss, axis=1)
        actions = np.stack(actions, axis=1)
        masks = np.stack(masks, axis=1)
        rewards = np.stack(rewards, axis=1)
        vals, counts = np.unique(rewards, return_counts=True)

        d = {float(v): int(c) for v, c in zip(vals, counts)}
        t = sum(d.values())
        pct = {k: c / t for k, c in d.items()}
        
        policies.append(agent.q.copy())
        G = agent.trajectory_update(actions, obss, rewards, masks)

        mean_rewards.append(G.mean())
        # print(d)
        history.append(pct)
        # print(pct)
        print(G.mean())

    plot_category_pcts(history[:], categories=[-1.0, 1.0, 1.5])

    
# import numpy as np



# example
# plot_category_pcts(history, categories=[-1.0, 0.0, 1.0, 1.5])

    


if __name__ == "__main__":
    main()