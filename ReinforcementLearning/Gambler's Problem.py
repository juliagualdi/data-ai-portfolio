import numpy as np
import matplotlib.pyplot as plt

# Gambler's Problem (Sutton & Barto, Ex. 4.3)
ph = 0.4                   # probability of heads
states = range(1, 100)     # nonterminal states
V = np.zeros(101, float)   # V[0]=V[100]=0 terminals
gamma = 1.0
theta = 1e-9            # smaller threshold is fine

def next_states(s, a):
    # returns (next_state, probability)
    return [
        (s+a, ph),  # head
        (s-a, 1-ph)  # tails
    ]

def expected_value(s, a, Vref):
    """One-step lookahead using reference values Vref (synchronous backup)."""
    e = 0.0
    for snext, prob in next_states(s, a):
        # reward is 1 if action leads to goal of 100 = s+a,
        # which is the sum of capital (s) and stake (a)
        # terminate if snext = 0 or 100 (dummy states for termination)
        reward = 1.0 if snext == 100 else 0.0
        e += prob * (reward + gamma * Vref[snext])

    return e

Delta = np.inf
iters = 0
while Delta > theta:
    Delta = 0.0
    Vk = V.copy()  # read from Vk, write into V
    for s in states:
         # possible stakes: 1..min(s, 100-s)
        stakes = range(1, min(s, 100 - s) + 1)
        # compute value for each action a (one-step lookahead) and pick max
        action_values = [expected_value(s, a, Vk) for a in stakes]
        best = max(action_values) if action_values else 0.0
        V[s] = best
        Delta = max(Delta, abs(V[s] - Vk[s]))
    iters += 1


print(f"Converged in {iters} sweeps")
plt.figure()
plt.plot(range(1, 100), V[1:100])
plt.title("Optimal Value Function V(s)")
plt.xlabel("Capital s"); plt.ylabel("V(s)")

def policy(s, Vref):
    if s == 0 or s == 100:
        return 0
    A = range(1, min(s, 100 - s) + 1)  # actions (starting at 1)
    vals = [expected_value(s, a, Vref) for a in A]
    return A[int(np.argmax(vals))]

final_policy = [policy(s, V) for s in states]
plt.figure()
plt.bar(list(states), final_policy, align='center', alpha=0.7)
plt.title("Optimal Policy (stake)"); plt.xlabel("Capital s"); plt.ylabel("Stake a")
plt.show()
