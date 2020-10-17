import gym
import numpy as np

from keras import initializers, optimizers
from keras.models import Sequential
from keras.layers import Dense

from keras.utils.np_utils import to_categorical

class AC():
    def __init__(
            self,
            actions_n,
            states_n,
            actor_learning_rate=0.001,
            critic_learning_rate=0.01,
            gamma=0.9):
        self.actions_n = actions_n
        self.states_n = states_n
        self.actor_lr = actor_learning_rate
        self.critic_lr = critic_learning_rate
        self.gamma = gamma

        self.__init_net()

    def __init_net(self):
        weight_init = initializers.RandomNormal(0, 0.1)
        bias_init = initializers.Constant(0.1)

        self.actor = Sequential()
        self.actor.add(Dense(20, input_dim=self.states_n, activation='relu', kernel_initializer=weight_init, bias_initializer=bias_init))
        self.actor.add(Dense(self.actions_n, activation='softmax', kernel_initializer=weight_init, bias_initializer=bias_init))
        self.actor.compile(optimizer=optimizers.Adam(self.actor_lr), loss='categorical_crossentropy')

        self.critic = Sequential()
        self.critic.add(Dense(20, input_dim=self.states_n, activation='relu', kernel_initializer=weight_init, bias_initializer=bias_init))
        self.critic.add(Dense(1, kernel_initializer=weight_init, bias_initializer=bias_init))
        self.critic.compile(optimizer=optimizers.Adam(self.critic_lr), loss='mse')

    def learn(self, state, action, reward, next_state):
        state = state.reshape(-1, self.states_n)
        action = to_categorical(action, self.actions_n).reshape(-1, self.actions_n)
        next_state = next_state.reshape(-1, self.states_n)

        td_error, critic_loss = self.__critic_learn(state, reward, next_state)
        actor_loss = self.actor.train_on_batch(state, action, sample_weight=td_error.reshape(-1, ))
        return critic_loss, actor_loss

    def __critic_learn(self, state, reward, next_state):
        V_now = self.critic.predict_on_batch(state)
        V_next = self.critic.predict_on_batch(next_state)
        y = reward + self.gamma * V_next
        td_error = y - V_now
        loss = self.critic.train_on_batch(state, y)
        return td_error, loss
    
    def choose_action(self, state):
        action_proba = self.actor.predict_on_batch(state.reshape(1, -1))
        action = np.random.choice(range(self.actions_n), p=action_proba.reshape(-1, ))
        return action


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    env = env.unwrapped
    RL = AC(env.action_space.n, env.observation_space.shape[0], 0.002, 0.01)

    sum_step = 0
    r1_weight = 0.3
    r2_weight = 1 - r1_weight
    pos_threshold = env.observation_space.high[0] / 2
    angle_threshold = env.observation_space.high[2] / 2
    for episode in range(200):
        state = env.reset()

        Reward = 0
        loss = []
        while True:
            env.render()
            action = RL.choose_action(state)
            next_state, reward, done, _ = env.step(action)

            x, x_dot, theta, theta_dot = next_state
            r1 = ((pos_threshold - abs(x)) / pos_threshold) * r1_weight
            r2 = ((angle_threshold - abs(theta)) / angle_threshold) * r2_weight
            reward = (r1 + r2) * 2 - 1

            loss.append(RL.learn(state, action, reward, next_state))
            state = next_state

            Reward += 1
            if done:
                sum_step += Reward
                loss = np.array(loss).reshape(-1, 2)
                print("\nepisode: {:>5}  , reward: {:>7}  , average reward: {:>10.5f}  in {:>10}  steps, critic loss: {:>10.5f}  , actor loss: {:>10.5f}".format(episode+1, Reward, sum_step / (episode+1), sum_step, np.mean(loss[:, 0]), np.mean(loss[:, 1])))
                break

    env.close()
