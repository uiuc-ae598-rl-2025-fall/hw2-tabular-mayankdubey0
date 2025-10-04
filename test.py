import numpy as np
import gymnasium as gym
from collections import defaultdict

def epsilon_greedy_action(Q, state, nA, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(nA)
    return int(np.argmax(Q[state]))

def generate_episode(env, Q, nA, epsilon, max_steps=200):
    states, actions, rewards = [], [], []
    s, _ = env.reset()
    for _ in range(max_steps):
        a = epsilon_greedy_action(Q, s, nA, epsilon)
        s_next, r, terminated, truncated, _ = env.step(a)
        states.append(s); actions.append(a); rewards.append(r)
        s = s_next
        if terminated or truncated:
            break
    return states, actions, rewards

def mc_control_on_policy(env, num_episodes=100_000, gamma=0.95,
                         epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.999):
    nS = env.observation_space.n
    nA = env.action_space.n
    Q = np.zeros((nS, nA), dtype=np.float64)
    returns_sum = np.zeros((nS, nA), dtype=np.float64)
    returns_cnt = np.zeros((nS, nA), dtype=np.int64)

    epsilon = epsilon_start
    for ep in range(num_episodes):
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        states, actions, rewards = generate_episode(env, Q, nA, epsilon)
        G = 0.0
        seen = set()  # for first-visit

        # work backwards computing returns
        for t in reversed(range(len(states))):
            s, a, r = states[t], actions[t], rewards[t]
            G = gamma * G + r
            if (s, a) in seen:
                continue
            seen.add((s, a))

            returns_sum[s, a] += G
            returns_cnt[s, a] += 1
            Q[s, a] = returns_sum[s, a] / returns_cnt[s, a]

    # derive a greedy policy from Q
    pi = np.argmax(Q, axis=1)
    return Q, pi

# ----- Example usage -----
if __name__ == "__main__":
    gamma = 0.95

    # Slippery
    env_slip = gym.make("FrozenLake-v1", render_mode=None,
                        desc=["SFFF","FHFH","FFFH","HFFG"], is_slippery=True)
    Q_s, pi_s = mc_control_on_policy(env_slip, num_episodes=80_000, gamma=gamma)

    # Non-slippery
    env_noslip = gym.make("FrozenLake-v1", render_mode=None,
                          desc=["SFFF","FHFH","FFFH","HFFG"], is_slippery=False)
    Q_ns, pi_ns = mc_control_on_policy(env_noslip, num_episodes=10_000, gamma=gamma)

    # print("Greedy policy (slippery):", pi_s.reshape(4,4))
    print("Greedy policy (non-slippery):", pi_ns)
