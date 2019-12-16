import numpy as np
import random
import sys
from collections import deque
import torch
import torch.optim as optim
from replay_buffer import ReplayBuffer
from params import DQNParameters

from model import QNetModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# Modified from the Udacity Deep Reinforcement Learning Nanodegree course materials.
# Added options for modifying the Q learning, such as double DQN, dueling DQN, and
# prioritized experience replay.
class DQNAgent:
    def __init__(self, params, model_name='temp.pth'):
        self.params = params
        self.model_name = model_name
        self.target_update_counter = 0

        # Q-Network
        self.qnetwork_local = QNetModel(self.params.state_size, self.params.action_size, self.params.seed,
                                        hidden_layers=self.params.hidden_layers, duelling=self.params.duelling).to(device)
        print(self.qnetwork_local)
        self.qnetwork_target = QNetModel(self.params.state_size, self.params.action_size, self.params.seed,
                                         hidden_layers=self.params.hidden_layers, duelling=self.params.duelling).to(device)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.params.learning_rate)

        # Replay memory
        self.memory = ReplayBuffer(self.params.replay_buffer_size, self.params.replay_batch_size, self.params.seed,
                                   prioritized=self.params.prioritized_replay_buffer, alpha=self.params.alpha)

        # Initialize the time step counter for updating each params.learn_every number of steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # reward clipping
        if self.params.reward_clipping:
            reward = np.clip(reward, -1.0, 1.0)

        if self.params.prioritized_replay_buffer:
            # Get max predicted Q values (for next states) from target model
            # Get index to action with maximum value in local network (instead of from target network - double_dqn))
            next_state_index = self.qnetwork_local(torch.FloatTensor(next_state).to(device)).data
            max_next_state_index = torch.argmax(next_state_index)
            # Get action value from target network
            next_state_Qvalue = self.qnetwork_target(torch.FloatTensor(next_state).to(device)).data
            max_next_state_Qvalue = next_state_Qvalue[max_next_state_index]

            target = reward + self.params.gamma() * max_next_state_Qvalue * (1 - done)
            old = self.qnetwork_local(torch.FloatTensor(state).to(device)).data[action]
            error = abs(old.item() - target.item())

            # TD error clipping
            if self.params.error_clipping and error > 1:
                error = 1

            # Save experience in replay memory
            self.memory.add(state, action, reward, next_state, done, error)
        else:
            self.memory.add(state, action, reward, next_state, done)

        # Learn every params.learn_every time steps.
        self.t_step = (self.t_step + 1) % self.params.learn_every
        if self.t_step == 0:
            # If enough samples are available in memory, get a random subset from the
            # saved experiences (weighted if prior_replay = True) and learn
            if len(self.memory) > self.params.replay_batch_size:
                experiences = self.memory.sample()
                self.learn(experiences)

    def act(self, state, epsilon=0.0):
        """
        Returns an action for a given state.
        state: current state
        epsilon (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action
        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.params.action_size))

    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
        """
        if self.params.prioritized_replay_buffer:
            states, actions, rewards, next_states, dones, priorities = experiences
        else:
            states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        # Get index to action with maximum value in local network (instead of from target network - double_dqn))
        next_state_local = self.qnetwork_local(next_states).detach()
        max_next_state_indices = torch.max(next_state_local, 1)[1]
        # Get action value from target network
        next_state_Qvalues = self.qnetwork_target(next_states).detach()
        max_next_state_Qvalues = next_state_Qvalues.gather(1, max_next_state_indices.unsqueeze(1))

        # Target Q values for current states
        target_Qvalues = rewards + self.params.gamma() * max_next_state_Qvalues * (1 - dones)

        # Predicted Q values from local model
        predicted_Qvalues = self.qnetwork_local(states).gather(1, actions)

        # Compute errors and loss. Clip the errors so they are between -1 and 1.
        errors = target_Qvalues - predicted_Qvalues
        if self.params.error_clipping:
            torch.clamp(errors, min=-1, max=1)

        if self.params.prioritized_replay_buffer:
            # beta = 1.0
            Pi = priorities / priorities.sum()
            wi = (1.0 / self.params.replay_buffer_size / Pi) ** self.params.beta()
            # Normalize wi as per Schaul et al., Prioritized Replay, ICLR 2016, https://arxiv.org/abs/1511.05952
            wi = wi/max(wi)
            errors *= wi
        loss = (errors ** 2).mean()

        # loss = F.mse_loss(predicted_Qvalues, target_Qvalues)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if self.params.gradient_clipping:
            torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), 1)

        self.optimizer.step()

        # Update target network
        if self.params.target_update_interval >= 0:
            # every target_update_interval times copy the entire local network to the target network
            if self.target_update_counter % self.params.target_update_interval == 0:
                self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
            self.target_update_counter += 1
        else:
            # alternatively make minor updates to target network in the direction of the local network.
            self.soft_update()

    # A gradual update of the target network used in the implementation of DQN
    # provided by the Udacity course materials for the deep reinforcement learning nanodegree
    # Basically, a weighted average with a small weight for the local_model
    def soft_update(self):
        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            target_param.data.copy_(self.params.tau() * local_param.data + (1.0 - self.params.tau()) * target_param.data)

    def env_reset(self, train_mode=True):
        raise Exception('env_reset method should be overridden.')

    def env_step(self, action, train_mode=True):
        raise Exception('env_step method should be overridden.')

    def train(self, n_episodes=2000, max_t=100000, passing_score=sys.maxsize):
        passing_score_achieved = False
        scores = []  # List of scores for an episode
        scores_window = deque(maxlen=100)  # Most recent 100 scores

        for i_episode in range(1, n_episodes + 1):
            # Reset environment
            state = self.env_reset(True)
            score = 0
            for t in range(max_t):

                # Get action
                action = self.act(state, self.params.epsilon())

                # Send action to environment
                next_state, reward, done = self.env_step(action, True)

                # run agent step - which adds to the experience and also learns based on UPDATE_EVERY
                self.step(state, action, reward, next_state, done)

                score += reward
                state = next_state

                # Exit if episode finished
                if done:
                    break

            scores_window.append(score)
            scores.append(score)

            self.params.epsilon.decrement()  # reduce randomness for action selection
            self.params.gamma.increment()  # give more priority to future returns
            self.params.tau.increment()  # make larger updates to target DQN
            self.params.beta.increment()  # increase bias for weights from stored experiences

            if i_episode % scores_window.maxlen == 0:
                print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.4f}\t{self.params}')
            else:
                print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.4f}\t{self.params}', end="")

            if np.mean(scores_window) >= passing_score and not passing_score_achieved:
                passing_score_achieved = True
                print(f'\rPassing score achieved in {i_episode} episodes!\tAverage Score: {np.mean(scores_window):.4f}')

        torch.save(self.qnetwork_local.state_dict(), self.model_name)

        return scores

    def test(self, n_episodes=100, max_t=sys.maxsize):
        """
        agent: the learning agent
        checkpoint: the saved weights for the pretrained neural network
        n_episodes: the number of episodes to run
        """
        # load the weights from file
        self.qnetwork_local.load_state_dict(torch.load(self.model_name))
        total_score = 0

        for i in range(n_episodes):

            # Reset the environment. Training mode is off.
            state = self.env_reset(False)
            score = 0

            for t in range(max_t):
                # Decide on an action given the current state
                action = self.act(state)

                # Send action to environment
                next_state, reward, done = self.env_step(action, False)

                # Add the current reward into the score
                score += reward
                state = next_state

                # Exit the loop when the episode is done
                if done:
                    break

            total_score += score
            print(f'Episode {i} Score: {score}')

        print(f'Average Score: {total_score / n_episodes}')
