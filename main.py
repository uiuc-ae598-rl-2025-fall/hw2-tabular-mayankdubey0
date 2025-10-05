import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

slippery = False

env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=slippery)

# Defined MDP:
P = env.unwrapped.P # transition matrix

# actions: 0=Left, 1=Down, 2=Right, 3=Up
A = [0, 1, 2, 3]
A_dict = {0:"left", 1:"down", 2:"right", 3:"up"}
nA = len(A)

# 4x4 grid 
S = [0, 1, 2, 3, 
     4, 5, 6, 7, 
     8, 9, 10, 11, 
     12, 13, 14, 15]
nS = len(S)

state, info = env.reset()

def MC_control(b_0=None, gamma=0.95, epsilon_s=0.3, max_iter=200, episodes=1000, eps_decay=0.99):
    Q = np.random.rand(nS, nA)
    s_a_count = np.zeros((nS, nA))

    # for plotting
    returns = []
    times = []
    mean_returns = []
    time_steps = 0
    
    epsilon = epsilon_s

    for ep in range(episodes):
        s, _ = env.reset()
        print(time_steps)
        if b_0 is not None:
            env.unwrapped.s = int(b_0)
            s = int(b_0)

        states, actions, rewards = [], [], []

        for t in range(max_iter):
            time_steps +=1
            if np.random.rand() < epsilon:
                a = np.random.randint(nA)
            else:
                a = np.argmax(Q[s])
            epsilon = epsilon * 0.9

            states.append(s)
            actions.append(a)

            s_next, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            rewards.append(r)
            s = s_next
            # time_steps += 1

            if done:
                # print("Reward:", r)
                break

        G = 0.0
        visited = set()
        for t in reversed(range(len(states))):
            G = rewards[t] + gamma * G
            sa = (states[t], actions[t])
            if sa not in visited:
                visited.add(sa)
                s_a_count[sa] += 1.0
                alpha = 0.2 
                Q[sa] += alpha * (G - Q[sa])
        time_steps += len(states)
        times.append(time_steps)
        returns.append(G)
        mean_returns.append(np.mean(returns))

        pi = np.argmax(Q, axis=1)

        epsilon = epsilon * eps_decay

    pi = np.argmax(Q, axis=1)
    return Q, pi, mean_returns, times


def SARSA(b_0, gamma=0.95, alpha=0.2, epsilon_s=0.1, max_iter=200):
    Q = np.random.rand(nS, nA)
    epsilon = epsilon_s
    returns = []
    times = []
    mean_returns = []
    time_steps = 0

    for i in range(1000):
        # print("episode:",i)
        states = []
        actions = []
        rewards = []

        s, _ = env.reset()
        env.unwrapped.s = b_0
        s = b_0   

        if np.random.rand() < epsilon:
            a = np.random.randint(nA)
        else:                              
            a = np.argmax(Q[s])
        epsilon = epsilon * 0.9

        for t in range(max_iter):

            time_steps += 1

            states.append(s)
            actions.append(a)

            s_prime, r, done, _, _ = env.step(a)
            rewards.append(r)

            sa = (states[t], actions[t])

            if np.random.rand() < epsilon:
                a_prime = np.random.randint(nA)
            else:                              
                a_prime = np.argmax(Q[s_prime])
            epsilon = epsilon * 0.9

            sa_p = (s_prime, a_prime)

            if done:
                Q[sa] += alpha*(rewards[t] - Q[sa])
                break
            
            Q[sa] += alpha*(rewards[t]+gamma*Q[sa_p] - Q[sa])

            s = s_prime
            a = a_prime

        G = 0.0
        for r in reversed(rewards):
            G = r + gamma * G
        returns.append(G)
        mean_returns.append(np.mean(returns))
        times.append(time_steps)

        pi = np.argmax(Q, axis=1)
    return Q, pi, mean_returns, times


def Q_learning(b_0, gamma=0.95, alpha=0.2, epsilon_s=0.1, max_iter=200):
    Q = np.random.rand(nS, nA)
    epsilon = epsilon_s
    returns = []
    times = []
    mean_returns = []
    time_steps = 0

    for i in range(1000):
        print("episode:",i)
        states = []
        actions = []
        rewards = []

        s, _ = env.reset()
        env.unwrapped.s = b_0
        s = b_0   

        if np.random.rand() < epsilon:
            a = np.random.randint(nA)
        else:                              
            a = np.argmax(Q[s])
        epsilon = epsilon * 0.9

        for t in range(max_iter):

            time_steps += 1

            states.append(s)
            actions.append(a)

            s_prime, r, done, _, _ = env.step(a)
            rewards.append(r)

            sa = (states[t], actions[t])

            if np.random.rand() < epsilon:
                a_prime = np.random.randint(nA)
            else:                              
                a_prime = np.argmax(Q[s_prime])
            epsilon = epsilon * 0.9

            sa_p = (s_prime, a_prime)

            if done:
                Q[sa] += alpha*(rewards[t] - Q[sa])
                break
            
            Q[sa] += alpha*(rewards[t]+gamma*np.max(Q[s_prime]) - Q[sa])

            s = s_prime
            a = a_prime

        G = 0.0
        for r in reversed(rewards):
            G = r + gamma * G
        returns.append(G)
        mean_returns.append(np.mean(returns))
        times.append(time_steps)

        pi = np.argmax(Q, axis=1)
    return Q, pi, mean_returns, times


print("Running MC...")
Q_MC, pi_MC, r_MC, t_MC  = MC_control(0)
print(Q_MC)
print("MC Policy:", pi_MC,"\n")

print("Running SARSA...")
Q_SARSA, pi_SARSA, r_SARSA, t_SARSA = SARSA(0)
print(Q_SARSA)
print("SARSA Policy:", pi_SARSA,"\n")

print("Running Q...")
Q_Q, pi_Q, r_Q, t_Q = Q_learning(0)
print(Q_Q)
print("Q Policy:", pi_Q,"\n")

# --- plotting ---
plt.figure(figsize=(7,4))
plt.plot(t_MC, r_MC, label='MC Control')
plt.plot(t_SARSA, r_SARSA, label='SARSA')
plt.plot(t_Q, r_Q, label='Q-Learning')

plt.title(f'FrozenLake 4x4 (slippery={slippery}): Learning Curves')
plt.xlabel('Time steps')
plt.ylabel('Mean return (running average)')
plt.legend(loc='best', frameon=True)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

slippery = True
print("Running MC...")
Q_MC, pi_MC, r_MC, t_MC  = MC_control(0)
print(Q_MC)
print("MC Policy:", pi_MC,"\n")

print("Running SARSA...")
Q_SARSA, pi_SARSA, r_SARSA, t_SARSA = SARSA(0)
print(Q_SARSA)
print("SARSA Policy:", pi_SARSA,"\n")

print("Running Q...")
Q_Q, pi_Q, r_Q, t_Q = Q_learning(0)
print(Q_Q)
print("Q Policy:", pi_Q,"\n")

# --- plotting ---
plt.figure(figsize=(7,4))
plt.plot(t_MC, r_MC, label='MC Control')
plt.plot(t_SARSA, r_SARSA, label='SARSA')
plt.plot(t_Q, r_Q, label='Q-Learning')

plt.title(f'FrozenLake 4x4 (slippery={slippery}): Learning Curves')
plt.xlabel('Time steps')
plt.ylabel('Mean return (running average)')
plt.legend(loc='best', frameon=True)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()