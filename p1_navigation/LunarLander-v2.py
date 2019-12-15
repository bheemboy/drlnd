import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from agent import Agent
from params import VarParam, DQNParameters


def dqn_train(agent, env, params, n_episodes=2000, max_t=100000):
    scores = []  # List of scores for an episode
    scores_window = deque(maxlen=100)  # Most recent 100 scores

    for i_episode in range(1, n_episodes + 1):
        # Reset environment
        state = env.reset()

        score = 0
        for t in range(max_t):

            # Get action
            action = agent.act(state, params.epsilon())

            # Send action to environment
            # env.render()
            next_state, reward, done, _ = env.step(action)

            # run agent step - which adds to the experience and also learns based on UPDATE_EVERY
            agent.step(state, action, reward, next_state, done)

            score += reward
            state = next_state

            # Exit if episode finished
            if done:
                break

        scores_window.append(score)
        scores.append(score)

        params.epsilon.decrement()  # reduce randomness for action selection
        params.gamma.increment()  # give more priority to future returns
        params.tau.increment()  # make larger updates to target DQN
        params.beta.increment()  # increase bias for weights from stored experiences

        print('\rEpisode {}\tAverage Score: {:.4f}\t{}'.format(i_episode, np.mean(scores_window), params), end="")

        if i_episode % scores_window.maxlen == 0:
            print('\rEpisode {}\tAverage Score: {:.4f}\t{}'.format(i_episode, np.mean(scores_window), params))
        else:
            print('\rEpisode {}\tAverage Score: {:.4f}\t{}'.format(i_episode, np.mean(scores_window), params), end="")

    torch.save(agent.qnetwork_local.state_dict(), 'lunarlander.pth')

    return scores


def load_and_run_agent(agent, env, n_episodes):
    """
    agent: the learning agent
    checkpoint: the saved weights for the pretrained neural network
    n_episodes: the number of episodes to run
    """

    # load the weights from file
    agent.qnetwork_local.load_state_dict(torch.load('lunarlander.pth'))

    total_score = 0

    for i in range(n_episodes):

        # Reset the environment. Training mode is off.
        state = env.reset()
        score = 0
        while True:
            # Decide on an action given the current state
            action = agent.act(state)

            # Send action to environment
            env.render()
            next_state, reward, done, _ = env.step(action)

            # Add the current reward into the score
            score += reward

            state = next_state

            # Exit the loop when the episode is done
            if done:
                break
        total_score += score

        print("Score: {}".format(score))

    print("Average Score: {}".format(total_score / n_episodes))


def main(train_mode=True):
    plt.ion()

    env = gym.make('LunarLander-v2')
    env.seed(0)

    # number of actions
    action_size = env.action_space.n
    print('Number of actions:', action_size)

    # examine the state space
    state_size = env.observation_space.shape[0]
    print('States have length:', state_size)

    params = DQNParameters(state_size=state_size, action_size=action_size)
    agent = Agent(params)

    if train_mode:
        scores = dqn_train(agent, env, params, n_episodes=2000)
        # plot the scores
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(len(scores)), scores)
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.show()
    else:
        load_and_run_agent(agent, env, n_episodes=100)


main(train_mode=False)
