import gym
import math
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F


class Agent(nn.Module):
    def __init__(self, s_size, a_size, h_size=8):
        super(Agent, self).__init__()
        # state, hidden layer, action sizes
        self.s_size = s_size
        self.h_size = h_size
        self.a_size = a_size
        # define layers
        self.fc1 = nn.Linear(self.s_size, self.h_size)
        self.fc2 = nn.Linear(self.h_size, self.a_size)

    def forward(self, state):
        x = F.relu(self.fc1(torch.from_numpy(state).float()))
        x = F.tanh(self.fc2(x))
        return x.cpu().data

    def act(self, state):
        probs = self.forward(state)
        # action = np.random.choice(2, p=probs) # option 1: stochastic policy
        action = np.argmax(probs)  # option 2: deterministic policy
        return action.numpy()

    def get_weights_dim(self):
        return (self.s_size + 1) * self.h_size + (self.h_size + 1) * self.a_size   # +1 is the bias

    def set_weights(self, weights):
        s_size = self.s_size
        h_size = self.h_size
        a_size = self.a_size
        # separate the weights for each layer
        fc1_end = (s_size * h_size) + h_size
        fc1_W = torch.from_numpy(weights[:s_size * h_size].reshape(s_size, h_size))
        fc1_b = torch.from_numpy(weights[s_size * h_size:fc1_end])
        fc2_W = torch.from_numpy(weights[fc1_end:fc1_end + (h_size * a_size)].reshape(h_size, a_size))
        fc2_b = torch.from_numpy(weights[fc1_end + (h_size * a_size):])
        # set the weights for each layer
        self.fc1.weight.data.copy_(fc1_W.view_as(self.fc1.weight.data))
        self.fc1.bias.data.copy_(fc1_b.view_as(self.fc1.bias.data))
        self.fc2.weight.data.copy_(fc2_W.view_as(self.fc2.weight.data))
        self.fc2.bias.data.copy_(fc2_b.view_as(self.fc2.bias.data))

    def get_weights(self):
        s_size = self.s_size
        h_size = self.h_size
        a_size = self.a_size
        fc1_end = (s_size * h_size) + h_size
        weights = np.zeros(self.get_weights_dim())
        weights[:s_size * h_size] = np.ravel(self.fc1.weight.data)
        weights[s_size * h_size:fc1_end] = np.ravel(self.fc1.bias.data)
        weights[fc1_end:fc1_end + (h_size * a_size)] = np.ravel(self.fc2.weight.data)
        weights[fc1_end + (h_size * a_size):] = np.ravel(self.fc2.bias.data)

        return weights


def hill_climbing(env, agent, n_episodes=10000, max_t=1000, gamma=1.0, print_every=100, noise_scale=1e-2):
    """Implementation of hill climbing with adaptive noise scaling.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        gamma (float): discount rate
        print_every (int): how often to print average score (over last 100 episodes)
        noise_scale (float): standard deviation of additive noise
    """
    scores_deque = deque(maxlen=100)
    scores = []
    best_R = -np.Inf
    best_w = 1e-4 * np.random.randn(agent.get_weights_dim())
    agent.set_weights(best_w)

    for i_episode in range(1, n_episodes + 1):
        rewards = []
        state = env.reset()
        for t in range(max_t):
            action = agent.act(state)
            state, reward, done, _ = env.step(action)
            rewards.append(reward)
            if done:
                break
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        discounts = [gamma ** i for i in range(len(rewards) + 1)]
        R = sum([a * b for a, b in zip(discounts, rewards)])

        if R >= best_R:  # found better weights
            # update the best values
            best_R = R
            best_w = agent.get_weights()
            # decrease noise scale
            noise_scale = max(1e-3, noise_scale / 2)
        else:  # did not find better weights
            # increase noise scale
            noise_scale = min(2, noise_scale * 2)
        agent.set_weights(best_w + noise_scale * np.random.rand(*best_w.shape))

        if i_episode % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
        if np.mean(scores_deque) >= 200.0:
            print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                       np.mean(scores_deque)))
            agent.set_weights(best_w)
            break

    return scores


def main():
    env = gym.make('LunarLander-v2')
    env.seed(0)
    np.random.seed(0)

    print('observation space:', env.observation_space)
    print('action space:', env.action_space)

    agent = Agent(8, 4)

    scores = hill_climbing(env, agent, n_episodes=10000, max_t=1000, gamma=1.0, print_every=100, noise_scale=1e-2)

    state = env.reset()
    for t in range(2000):
        env.render()
        action = agent.act(state)
        state, reward, done, _ = env.step(action)
        if done:
            break

    env.close()


main()
