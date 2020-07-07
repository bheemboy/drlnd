import pong_utils
import gym
import time
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from parallelEnv import parallelEnv
import numpy as np
import progressbar as pb


if __name__ == '__main__':
    # check which device is being used.
    # I recommend disabling gpu until you've made sure that the code runs
    device = pong_utils.device
    print("using device: ", device)

    # render ai gym environment
    # PongDeterministic does not contain random frameskip
    # so is faster to train than the vanilla Pong-v4 environment
    env = gym.make('PongDeterministic-v4')

    print("List of available actions: ", env.unwrapped.get_action_meanings())

    # we will only use the actions 'RIGHTFIRE' = 4 and 'LEFTFIRE" = 5
    # the 'FIRE' part ensures that the game starts again after losing a life
    # the actions are hard-coded in pong_utils.py

    policy=pong_utils.Policy().to(device)

    # we use the adam optimizer with learning rate 2e-4
    # optim.SGD is also possible
    optimizer = optim.Adam(policy.parameters(), lr=1e-4)

    # training loop max iterations
    episode = 500

    widget = ['training loop: ', pb.Percentage(), ' ', pb.Bar(), ' ', pb.ETA()]
    timer = pb.ProgressBar(widgets=widget, maxval=episode).start()

    envs = parallelEnv('PongDeterministic-v4', n=8, seed=1234)

    discount_rate = .99
    epsilon = 0.1
    beta = .01
    tmax = 320
    SGD_epoch = 4

    # keep track of progress
    mean_rewards = []

    for e in range(episode):

        # collect trajectories
        old_probs, states, actions, rewards = pong_utils.collect_trajectories(envs, policy, tmax=tmax)

        total_rewards = np.sum(rewards, axis=0)

        # gradient ascent step
        for _ in range(SGD_epoch):
            L = -pong_utils.clipped_surrogate(policy, old_probs, states, actions, rewards,
                                              epsilon=epsilon, beta=beta)
            optimizer.zero_grad()
            L.backward()
            optimizer.step()
            del L

        # the clipping parameter reduces as time goes on
        epsilon *= .999

        # the regulation term also reduces
        # this reduces exploration in later runs
        beta *= .995

        # get the average reward of the parallel environments
        mean_rewards.append(np.mean(total_rewards))

        # display some progress every 20 iterations
        if (e + 1) % 20 == 0:
            print("Episode: {0:d}, score: {1:f}".format(e + 1, np.mean(total_rewards)))
            print(total_rewards)

        # update progress widget bar
        timer.update(e + 1)

    timer.finish()
