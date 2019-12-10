import numpy as np
import random
import torch
import torch.optim as optim
from replay_buffer import ReplayBufferPrioritized
from params import DQNParameters

from model import QNetDueling

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 64  # minibatch size
UPDATE_EVERY = 4  # Number of environment steps between every update with experience replay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Modified from the Udacity Deep Reinforcement Learning Nanodegree course materials.
# Added options for modifying the Q learning, such as double DQN, dueling DQN, and
# prioritized experience replay.
class Agent:
    def __init__(self, params):
        self.params = params
        self.target_update_counter = 0

        # Q-Network
        self.qnetwork_local = QNetDueling(self.params.state_size, self.params.action_size, self.params.seed).to(device)
        self.qnetwork_target = QNetDueling(self.params.state_size, self.params.action_size, self.params.seed).to(device)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.params.learning_rate)

        # Replay memory
        self.memory = ReplayBufferPrioritized(self.params.action_size, BUFFER_SIZE, BATCH_SIZE, self.params.seed, alpha=self.params.alpha)

        # Initialize the time step counter for updating each UPDATE_EVERY number of steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done, gamma=0.99, beta=0.0, tau=0.001):

        # reward clipping
        if self.params.reward_clipping:
            reward = np.clip(reward, -1.0, 1.0)

        # Get max predicted Q values (for next states) from target model
        # Get index to action with maximum value in local network (instead of from target network - double_dqn))
        next_state_index = self.qnetwork_local(torch.FloatTensor(next_state).to(device)).data
        max_next_state_index = torch.argmax(next_state_index)
        # Get action value from target network
        next_state_Qvalue = self.qnetwork_target(torch.FloatTensor(next_state).to(device)).data
        max_next_state_Qvalue = next_state_Qvalue[max_next_state_index]

        target = reward + gamma * max_next_state_Qvalue * (1 - done)
        old = self.qnetwork_local(torch.FloatTensor(state).to(device)).data[action]
        error = abs(old.item() - target.item())

        # TD error clipping
        if self.params.error_clipping and error > 1:
            error = 1

        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done, error)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get a random subset from the
            # saved experiences (weighted if prior_replay = True) and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, gamma, beta, tau)

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

    def learn(self, experiences, gamma=0.99, beta=0.0, tau=0.001):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, priorities = experiences

        # Get max predicted Q values (for next states) from target model
        # Get index to action with maximum value in local network (instead of from target network - double_dqn))
        next_state_local = self.qnetwork_local(next_states).detach()
        max_next_state_indices = torch.max(next_state_local, 1)[1]
        # Get action value from target network
        next_state_Qvalues = self.qnetwork_target(next_states).detach()
        max_next_state_Qvalues = next_state_Qvalues.gather(1, max_next_state_indices.unsqueeze(1))

        # Target Q values for current states
        target_Qvalues = rewards + gamma * max_next_state_Qvalues * (1 - dones)

        # Predicted Q values from local model
        predicted_Qvalues = self.qnetwork_local(states).gather(1, actions)

        # Compute errors and loss. Clip the errors so they are between -1 and 1.
        errors = target_Qvalues - predicted_Qvalues
        if self.params.error_clipping:
            torch.clamp(errors, min=-1, max=1)

        # beta = 1.0
        Pi = priorities / priorities.sum()
        wi = (1.0 / BUFFER_SIZE / Pi) ** beta
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
            self.soft_update(tau)

    # A gradual update of the target network used in the implementation of DQN
    # provided by the Udacity course materials for the deep reinforcement learning nanodegree
    # Basically, a weighted average with a small weight for the local_model
    def soft_update(self, tau=1e-3):
        """
        target_model = τ*local_model + (1 - τ) * target_model
        local_model (PyTorch model): weights will be copied from
        target_model (PyTorch model): weights will be copied to
        tau: a small weight for the local model
        """
        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
