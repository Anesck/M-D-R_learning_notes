import gym
import numpy as np

from keras import initializers, optimizers
from keras.models import Sequential
from keras.layers import Dense
from keras.utils.np_utils import to_categorical

class PG():
    def __init__(
            self,
            actions_n,
            states_n,
            learning_rate=0.01,
            gamma=0.95,     # reward decay
            ):
        self.actions_n = actions_n
        self.states_n = states_n
        self.lr = learning_rate
        self.gamma = gamma

        self.states_memory = []
        self.actions_memory = []
        self.reward_memory = []
        self.__init_net()

    def __init_net(self):
        weight_init = initializers.RandomNormal(0, 0.3)
        bias_init = initializers.Constant(0.1)
        model = Sequential()
        model.add(Dense(10, input_dim=self.states_n, activation='tanh', kernel_initializer=weight_init, bias_initializer=bias_init))
        model.add(Dense(self.actions_n, activation='softmax', kernel_initializer=weight_init, bias_initializer=bias_init))
        model.compile(optimizer=optimizers.Adam(self.lr), loss='categorical_crossentropy')
        self.actor = model

    def store_memory(self, state, action, reward):
        self.states_memory.append(state)
        self.actions_memory.append(action)
        self.reward_memory.append(reward)

    def choose_action(self, state):
        action_proba = self.actor.predict_on_batch(state.reshape(1, -1))
        action = np.random.choice(range(self.actions_n), p=action_proba.reshape(-1, ))
        return action

    def learn(self):
        discounted_rewards = self.__discount_rewards()
        x = np.vstack(self.states_memory)
        y = to_categorical(self.actions_memory, self.actions_n)

        self.states_memory = []
        self.actions_memory = []
        self.reward_memory = []
        loss =  self.actor.train_on_batch(x, y, sample_weight=discounted_rewards)
        return loss

    def __discount_rewards(self):
        discounted_rewards = np.zeros_like(self.reward_memory)
        accumulation_reward = 0
        for i in reversed(range(0, len(self.reward_memory))):
            accumulation_reward = accumulation_reward * self.gamma + self.reward_memory[i]
            discounted_rewards[i] = accumulation_reward

        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)
        return discounted_rewards


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    env = env.unwrapped
    RL = PG(env.action_space.n, env.observation_space.shape[0], 0.02)

    sum_step = 0
    for episode in range(500):
        state = env.reset()
        while True:
            env.render()
            action = RL.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            RL.store_memory(state, action, reward)
            state = next_state
            
            if done:
                Reward = sum(RL.reward_memory)
                loss = RL.learn()

                sum_step += Reward
                print("episode: {:<5}, reward: {:<10}, average reward: {:<20} in {} steps, train loss: {}".format(episode+1, Reward, sum_step / (episode+1), sum_step, loss))
                break

    env.close()



