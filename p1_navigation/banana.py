import numpy as np
import matplotlib.pyplot as plt
from agent import DQNAgent
import platform
from params import DQNParameters
from unityagents import UnityEnvironment


class UnityBanana(DQNAgent):
    def __init__(self):
        self.os = platform.system().lower()
        if self.os == 'linux':
            self.env = UnityEnvironment(file_name='Banana_Linux/Banana.x86_64')
        elif self.os == 'windows':
            self.env = UnityEnvironment(file_name='Banana_Windows_x86_64/Banana.exe')
        else:
            self.env = None

        # get the default brain
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]

        env_info = self.env.reset()[self.brain_name]

        # number of actions
        action_size = self.brain.vector_action_space_size
        print(f'Number of actions:{action_size}')

        # examine the state space
        state = env_info.vector_observations[0]
        state_size = len(state)
        print(f'States have length:{state_size}')

        params = DQNParameters(state_size=state_size, action_size=action_size)
        # params.prioritized_replay_buffer = False
        params.hidden_layers = [37, 16, 8]
        # params.duelling = False

        super(UnityBanana, self).__init__(params, model_name='UnityBanana.pth')

    def env_reset(self, train_mode=True):
        # Reset environment
        env_info = self.env.reset(train_mode=True)[self.brain_name]
        # Get next state
        state = env_info.vector_observations[0]
        return state

    def env_step(self, action, train_mode=True):
        if self.os == 'windows':
            env_info = self.env.step(action.astype(np.int32))[self.brain_name]
        else:
            env_info = self.env.step(action)[self.brain_name]
        next_state = env_info.vector_observations[0]    # Get next state
        reward = env_info.rewards[0]                    # Get reward
        done = env_info.local_done[0]                   # Check is episode finished
        return next_state, reward, done


dqn = UnityBanana()
scores = dqn.train(n_episodes=2000, passing_score=13)

# plot the scores
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

# dqn.test()
