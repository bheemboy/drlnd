import gym
import numpy as np
import matplotlib.pyplot as plt
from agent import DQNAgent
from params import DQNParameters


class LunarLander(DQNAgent):
    def __init__(self):
        self.env = gym.make('LunarLander-v2')
        self.env.seed(0)

        # number of actions
        action_size = self.env.action_space.n
        print(f'Number of actions:{action_size}')

        # examine the state space
        state_size = self.env.observation_space.shape[0]
        print(f'States have length:{state_size}')

        params = DQNParameters(state_size=state_size, action_size=action_size)
        # params.prioritized_replay_buffer = False
        # params.hidden_layers = [16, 8, 4]
        # params.duelling = False

        super(LunarLander, self).__init__(params, model_name='LunarLander-v2.pth')

    def env_reset(self, train_mode=True):
        return self.env.reset()

    def env_step(self, action, train_mode=True):
        if not train_mode:
            self.env.render()
        next_state, reward, done, _ = self.env.step(action)
        return next_state, reward, done


dqn = LunarLander()
scores = dqn.train(passing_score=200)

# plot the scores
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

# dqn.test()
