import random
from collections import namedtuple, deque
import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    """
    Prioritized experience replay.
    Adds priority to the stored experience tuples.
    Weighted random sampling of the experiences.
    """
    def __init__(self, buffer_size, batch_size, seed=0, prioritized=True, alpha=0.6, e_priority=0.01):
        """
        buffer_size (int): maximum size of buffer
        batch_size (int): size of each training batch
        seed (int): random seed
        prioritized (bool): prioritized replay buffer
        alpha (float): exponent for computing priorities
        e_priority (float): small additive factor to ensure that no experience has 0 priority
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.prioritized = prioritized
        if self.prioritized:
            self.experience = namedtuple("Experience",
                                         field_names=["state", "action", "reward", "next_state", "done", "priority"])
        else:
            self.experience = namedtuple("Experience",
                                         field_names=["state", "action", "reward", "next_state", "done"])
        self.alpha = alpha
        self.e_priority = e_priority

    def add(self, state, action, reward, next_state, done, error=None):
        """Add a new experience to memory."""
        #  Prioritize experiences where there is a big difference between our prediction and the TD target
        #  since it means that we have a lot to learn about it.
        if self.prioritized:
            priority = (error + self.e_priority) ** self.alpha
            self.memory.append(self.experience(state, action, reward, next_state, done, priority))
        else:
            self.memory.append(self.experience(state, action, reward, next_state, done))

    def sample(self):
        """
        Weighted random sampling experiences in memory.
        """
        if self.prioritized:
            priorities = [e.priority for e in self.memory if e is not None]
            experiences = random.choices(population=self.memory, weights=priorities, k=self.batch_size)
        else:
            experiences = random.choices(population=self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        if self.prioritized:
            priorities = torch.from_numpy(np.vstack([e.priority for e in experiences if e is not None])).float().to(device)
            return states, actions, rewards, next_states, dones, priorities
        else:
            return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
