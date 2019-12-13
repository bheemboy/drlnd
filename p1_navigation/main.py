import platform
from collections import deque
import numpy as np
import torch
import matplotlib.pyplot as plt

from unityagents import UnityEnvironment
from agent import Agent
from params import VarParam, DQNParameters

os = platform.system().lower()


def dqn_train(agent, env, params, n_episodes=2000, max_t=100000):
    brain_name = env.brain_names[0]
    passing_score_achieved = False  # Achieved when average score for 100 episodes reaches 13
    scores = []  # List of scores for an episode
    scores_window = deque(maxlen=100)  # Most recent 100 scores

    for i_episode in range(1, n_episodes + 1):
        # Reset environment
        env_info = env.reset(train_mode=True)[brain_name]

        # Get next state
        state = env_info.vector_observations[0]

        score = 0
        for t in range(max_t):

            # Get action
            action = agent.act(state, params.epsilon())

            # Send action to environment
            if os == 'windows':
                env_info = env.step(action.astype(np.int32))[brain_name]
            else:
                env_info = env.step(action)[brain_name]

            # Get next state
            next_state = env_info.vector_observations[0]

            # Get reward
            reward = env_info.rewards[0]

            # Check is episode finished
            done = env_info.local_done[0]

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
            print('\r\nEpisode {}\tAverage Score: {:.4f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) > 13 and not passing_score_achieved:
            passing_score_achieved = True
            print('\rPassing score achieved in {:d} episodes!\tAverage Score: {:.4f}'.format(i_episode,
                                                                                             np.mean(scores_window)))
    if passing_score_achieved:
        torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
    else:
        print("Passing score not achieved :(")

    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

    return scores


def load_and_run_agent(agent, env, n_episodes):
    """
    agent: the learning agent
    checkpoint: the saved weights for the pretrained neural network
    n_episodes: the number of episodes to run
    """

    # load the weights from file
    agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))
    brain_name = env.brain_names[0]

    total_score = 0

    for i in range(n_episodes):

        # Reset the environment. Training mode is off.
        env_info = env.reset(train_mode=False)[brain_name]

        # Get the current state
        state = env_info.vector_observations[0]

        score = 0

        while True:
            # Decide on an action given the current state
            action = agent.act(state)

            # Send action to the environment
            if os == 'windows':
                env_info = env.step(action.astype(np.int32))[brain_name]
            else:
                env_info = env.step(action)[brain_name]

            # Get the next state
            next_state = env_info.vector_observations[0]

            # Get the reward
            reward = env_info.rewards[0]

            # Check if the episode is finished
            done = env_info.local_done[0]

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
    if os == 'linux':
        env = UnityEnvironment(file_name='Banana_Linux/Banana.x86_64')
    elif os == 'windows':
        env = UnityEnvironment(file_name='Banana_Windows_x86_64/Banana.exe')
    else:
        env = None

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=train_mode)[brain_name]

    # number of actions
    action_size = brain.vector_action_space_size
    print('Number of actions:', action_size)

    # examine the state space
    state = env_info.vector_observations[0]
    state_size = len(state)
    print('States have length:', state_size)

    params = DQNParameters(state_size=state_size, action_size=action_size)
    agent = Agent(params)

    if train_mode:
        scores = dqn_train(agent, env, params, n_episodes=2000)
    else:
        load_and_run_agent(agent, env, n_episodes=100)


main(train_mode=True)
