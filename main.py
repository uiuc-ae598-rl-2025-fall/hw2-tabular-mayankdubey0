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

def MC_policy_eval(pi, b_0=0, gamma=0.9, max_iter=10000):
    V = np.random(nS)

    states = []
    actions = []
    rewards = []
    state_count = np.zeros(nS)
    s = b_0

    for t in range(max_iter): # runs each episode
        a = pi[s] # choose action
        actions.append(a)
        s_prime, r, done, _, _ = env.step(a) # take action and move to next state
        states.append(s_prime)
        rewards.append(r) # add reward
        if done:
            break
        s = s_prime
        if s in states:
            state_count[s] += 1
    
    G = 0
    visited_states = set()
    for t in range(len(states)):
        G = rewards[t] + gamma*G
        if states[t] not in visited_states: # first visit
            V[states[t]] += (1/state_count[states[t]]) * (G - V(states[t]))
            states.add(states[t])


def MC_control(b_0, gamma=0.95, epsilon_s=0.1, max_iter=10000):
    Q = np.random.rand(nS, nA)
    s_a_count = np.zeros((nS, nA))

    for i in range(100):
        # print("episode:",i)
        states = []
        actions = []
        rewards = []
        epsilon = epsilon_s

        s, _ = env.reset()
        env.unwrapped.s = b_0
        s = b_0   

        for t in range(max_iter):
            if np.random.rand() < epsilon:
                a = np.random.randint(nA)
            else:                              
                a = np.argmax(Q[s])
            epsilon = epsilon * 0.9

            states.append(s)
            actions.append(a)

            s_prime, r, done, _, _ = env.step(a)
            rewards.append(r)

            if done:
                break
            s = s_prime

        G = 0.0
        visited_s_a = set()
        for t in reversed(range(len(states))):
            G = rewards[t] + gamma * G
            sa = (states[t], actions[t])
            if sa not in visited_s_a:
                visited_s_a.add(sa)
                s_a_count[sa] += 1
                Q[sa] += (G - Q[sa]) / s_a_count[sa]

        pi = np.argmax(Q, axis=1)
    return Q, pi


def SARSA(b_0, gamma=0.95, alpha=0.2, epsilon_s=0.1, max_iter=10000):
    Q = np.random.rand(nS, nA)

    for i in range(10000):
        # print("episode:",i)
        states = []
        actions = []
        rewards = []
        epsilon = epsilon_s

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

    for i in range(100):
        # print("episode:",i)
        states = []
        actions = []
        rewards = []
        epsilon = epsilon_s

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
            
            Q[sa] += alpha*(rewards[t]+gamma*np.argmax(Q[sa_p]) - Q[sa])

            s = s_prime
            a = a_prime

        pi = np.argmax(Q, axis=1)
    return Q, pi

# Q, pi = MC_control(0)
# Q, pi = SARSA(0)
Q, pi = Q_learning(0)

# print("Optimal Action Value Function:\n", Q, "\n\n")
print("Optimal Policy:\n", pi)


