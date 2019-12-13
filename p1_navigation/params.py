class VarParam:
    """
    Class to manage dqn parameters that need to incremented or decremented during the training
    """
    def __init__(self, start, end=None, rate=0.01):
        self.__start = float(start)
        self.__current_val = self.__start
        if end is None:
            self.__end = self.__start
        else:
            self.__end = float(end)
        self.__scale = 1.0 - rate

        self.__min = min(self.__start, self.__end)
        self.__max = max(self.__start, self.__end)

    def __call__(self):
        return self.__current_val

    def increment(self):
        if self.__max > self.__current_val:
            self.__current_val = self.__max - self.__scale * (self.__max - self.__current_val)
        return self.__current_val

    def decrement(self):
        if self.__min < self.__current_val:
            self.__current_val = self.__min + self.__scale * (self.__current_val - self.__min)
        return self.__current_val


class DQNParameters:
    """
    Class that holds DQN training parameters
    """
    def __init__(self,
                 state_size,                             # Size of state input from environment
                 action_size,                            # Number of actions that the agent can take
                 seed=0,                                 #
                 learning_rate=0.0005,                   # learning rate
                 replay_buffer_size=int(1e6),            # replay buffer size
                 replay_batch_size=64,                   # minibatch size
                 learn_every=4,                          # Number of environment steps between every update with experience replay
                 target_update_interval=-1.0,            # The number of learning steps between updating the neural network for fixed Q targets.
                                                         # Set negative to use soft updating instead.
                 alpha=0.6,                              # Exponent for computing priorities in replay buffer
                 error_clipping=False,                   # Flag for limiting the TD error to between -1 and 1
                 gradient_clipping=False,                # Flag for clipping the norm of the gradient to 1
                 reward_clipping=False,                   # Flag for limiting the reward to between -1 and 1
                 epsilon=VarParam(1.0, 0.01, 0.005),     # Epsilon-greedy selection of an action. Probably for selecting a random action from the QTable
                 gamma=VarParam(0.9, 0.99, 0.01),        # Discount factor (0 to 1). Higher values favor long term over current rewards.

                 beta=VarParam(0.4, 1.0, 0.005),         # For prioritized replay. Corrects bias induced by weighted sampling of stored experiences.
                                                         # increase to 1.0 as per Schauel et al. https://arxiv.org/abs/1511.05952
                 tau=VarParam(0.001, 0.1, 0.00001)       # weighting factor for soft updating the target network in DQN.
                 ):
        self.epsilon = epsilon
        self.gamma = gamma
        self.beta = beta
        self.tau = tau
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
        self.learning_rate = learning_rate
        self.replay_buffer_size = replay_buffer_size
        self.replay_batch_size = replay_batch_size
        self.learn_every = learn_every
        self.target_update_interval = target_update_interval
        self.alpha = alpha
        self.error_clipping = error_clipping
        self.gradient_clipping = gradient_clipping
        self.reward_clipping = reward_clipping

    def __str__(self):
        return "lr={}, epsilon={:.3f}, gamma={:.3f}, beta={:.3f}, tau={:.3f}".format(
            self.learning_rate, self.epsilon(), self.gamma(), self.beta(), self.tau())
