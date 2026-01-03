import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt

# Environment: Cliff Walking 4x12
class CliffEnv:
    def __init__(self, rows=4, cols=12):
        self.rows = rows
        self.cols = cols
        self.start = (rows-1, 0)
        self.goal = (rows-1, cols-1)
        self.cliff = {(self.rows-1, c) for c in range(1, cols-1)}
        self.n_states = rows * cols
        self.n_actions = 4  # up, down, left, right

    def state_to_idx(self, s):
        r, c = s
        return r * self.cols + c

    def reset(self):
        self.pos = self.start
        return self.state_to_idx(self.pos)

    def step(self, action):
        r, c = self.pos
        if action == 0: nr, nc = max(r-1, 0), c       # up
        elif action == 1: nr, nc = min(r+1, self.rows-1), c  # down
        elif action == 2: nr, nc = r, max(c-1, 0)     # left
        elif action == 3: nr, nc = r, min(c+1, self.cols-1)  # right
        else: raise ValueError("Invalid action")

        newpos = (nr, nc)
        reward, done = -1.0, False

        if newpos in self.cliff:
            reward, self.pos = -100.0, self.start
            return self.state_to_idx(self.pos), reward, done

        if newpos == self.goal:
            self.pos = newpos
            return self.state_to_idx(newpos), reward, True

        self.pos = newpos
        return self.state_to_idx(newpos), reward, done


# Epsilon-greedy
def epsilon_greedy(Q, s, n_actions, epsilon):
    if random.random() < epsilon:
        return random.randrange(n_actions)
    qvals = [Q[s, a] for a in range(n_actions)]
    maxv = max(qvals)
    bests = [i for i, v in enumerate(qvals) if v == maxv]
    return random.choice(bests)

# Algorithms
def run_sarsa(env, episodes=500, alpha=0.5, gamma=1.0, epsilon=0.1, seed=None):
    if seed is not None:
        random.seed(seed); np.random.seed(seed)
    Q = np.zeros((env.n_states, env.n_actions))
    rewards = []
    for ep in range(episodes):
        s = env.reset()
        a = epsilon_greedy(Q, s, env.n_actions, epsilon)
        done, total_r = False, 0.0
        while not done:
            ns, r, done = env.step(a)
            na = epsilon_greedy(Q, ns, env.n_actions, epsilon)
            Q[s, a] += alpha * (r + gamma * Q[ns, na] - Q[s, a])
            s, a = ns, na
            total_r += r
        rewards.append(total_r)
    return Q, rewards

def run_q_learning(env, episodes=500, alpha=0.5, gamma=1.0, epsilon=0.1, seed=None):
    if seed is not None:
        random.seed(seed); np.random.seed(seed)
    Q = np.zeros((env.n_states, env.n_actions))
    rewards = []
    for ep in range(episodes):
        s = env.reset()
        done, total_r = False, 0.0
        while not done:
            a = epsilon_greedy(Q, s, env.n_actions, epsilon)
            ns, r, done = env.step(a)
            Q[s, a] += alpha * (r + gamma * np.max(Q[ns]) - Q[s, a])
            s = ns
            total_r += r
        rewards.append(total_r)
    return Q, rewards

# Multi-run
def multi_run(algorithm_fn, env, episodes=500, runs=10, **kwargs):
    all_rewards = np.zeros((runs, episodes))
    last_Qs = []
    for run in range(runs):
        Q, rewards = algorithm_fn(env, episodes=episodes, seed=run, **kwargs)
        all_rewards[run] = rewards
        last_Qs.append(Q)
    mean_rewards = np.mean(all_rewards, axis=0)
    std_rewards = np.std(all_rewards, axis=0)
    return mean_rewards, std_rewards, last_Qs[-1]

def greedy_policy_from_Q(Q, env):
    policy = np.zeros(env.n_states, dtype=int)
    for s in range(env.n_states):
        policy[s] = int(np.argmax(Q[s]))
    return policy.reshape((env.rows, env.cols))

# Run experiments and plot
if __name__ == "__main__":
    env = CliffEnv()
    episodes = 500
    runs = 10

    print("Running SARSA...")
    mean_sarsa, std_sarsa, Qs = multi_run(
        run_sarsa,
        env,
        episodes=episodes,
        runs=runs,
        alpha=0.5,
        gamma=1.0,
        epsilon=0.1
    )

    print("Running Q-Learning...")
    mean_q, std_q, Qq = multi_run(
        run_q_learning,
        env,
        episodes=episodes,
        runs=runs,
        alpha=0.5,
        gamma=1.0,
        epsilon=0.1
    )

    # Plot mean rewards (with shading)
    plt.figure(figsize=(10, 5))
    x = np.arange(1, episodes + 1)
    plt.plot(x, mean_sarsa, label="SARSA (mean over runs)")
    plt.fill_between(x, mean_sarsa - std_sarsa, mean_sarsa + std_sarsa, alpha=0.2)
    plt.plot(x, mean_q, label="Q-Learning (mean over runs)")
    plt.fill_between(x, mean_q - std_q, mean_q + std_q, alpha=0.2)
    plt.xlabel("Episode")
    plt.ylabel("Sum of rewards in episode")
    plt.title("Cliff Walking: SARSA vs Q-Learning")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    # Print greedy policies for the last Q of each method
    print("\nGreedy policy (action indices) from SARSA (last run):")
    print(greedy_policy_from_Q(Qs, env))

    print("\nGreedy policy (action indices) from Q-Learning (last run):")
    print(greedy_policy_from_Q(Qq, env))
