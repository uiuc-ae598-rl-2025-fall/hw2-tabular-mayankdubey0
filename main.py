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

def plot_graph(time_steps, returns):
    plt.plot(time_steps, returns)
    plt.show()


def MC_control(b_0=None, gamma=0.95, epsilon_s=0.3, max_iter=200, episodes=10000, eps_decay=0.99):
    Q = np.random.rand(nS, nA)
    s_a_count = np.zeros((nS, nA))
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
                alpha = 0.2 #1.0 / s_a_count[sa]
                Q[sa] += alpha * (G - Q[sa])
        time_steps += len(states)
        times.append(time_steps)
        returns.append(G)
        mean_returns.append(np.mean(returns))

        pi = np.argmax(Q, axis=1)

        epsilon = epsilon * eps_decay

    pi = np.argmax(Q, axis=1)
    return Q, pi, mean_returns, times


def SARSA(b_0, gamma=0.95, alpha=0.2, epsilon_s=0.1, max_iter=10000):
    Q = np.random.rand(nS, nA)
    epsilon = epsilon_s

    for i in range(500):
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

        pi = np.argmax(Q, axis=1)
    return Q, pi


def Q_learning(b_0, gamma=0.95, alpha=0.2, epsilon_s=0.1, max_iter=10000):
    Q = np.random.rand(nS, nA)
    epsilon = epsilon_s

    for i in range(500):
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

            if done:
                Q[sa] += alpha*(rewards[t] - Q[sa])
                break
            
            Q[sa] += alpha*(rewards[t]+gamma*np.argmax(Q[s_prime]) - Q[sa])

            s = s_prime
            a = a_prime

        pi = np.argmax(Q, axis=1)
    return Q, pi

print("Running MC...")
Q_MC, pi_MC, r, t  = MC_control(0)
print(Q_MC)
plot_graph(t, r)
print("MC Policy:", pi_MC,"\n")
# print("Running SARSA...")
# Q_SARSA, pi_SARSA = SARSA(0)
# print("SARSA Policy:", pi_SARSA,"\n")
# print("Running Q...")
# Q_Q, pi_Q = Q_learning(0)
# print("Q Policy:", pi_Q,"\n")

# print("Optimal Action Value Function:\n", Q, "\n\n")



