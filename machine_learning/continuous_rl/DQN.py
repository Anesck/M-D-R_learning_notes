import gym
import numpy as np
import matplotlib.pyplot as plt

from keras import initializers, optimizers
from keras.models import Sequential, clone_model
from keras.layers import Dense

class DQN():
    def __init__(
            self,
            actions_n,
            states_n,
            learning_rate=0.01,
            gamma=0.9,
            epsilon=0.1,
            update_target_iter=200,
            memory_size=2000,
            batch_size=32,
            epsilon_decrement=False,
            ):
        self.actions_n = actions_n
        self.states_n = states_n
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon_min = epsilon
        self.update_target_iter = update_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_decrement = epsilon_decrement
        self.epsilon = 1 if epsilon_decrement else self.epsilon_min

        self.step_counter = 0
        self.memory = np.zeros([self.memory_size, states_n * 2 + 2])
        self.__init_net()

    def __init_net(self):
        weight_init = initializers.RandomNormal(0, 0.3)
        bias_init = initializers.Constant(0.1)
        model = Sequential()
        model.add(Dense(32, input_dim=self.states_n, activation='relu', kernel_initializer=weight_init, bias_initializer=bias_init))
        model.add(Dense(self.actions_n, kernel_initializer=weight_init, bias_initializer=bias_init))
        model.compile(optimizer=optimizers.RMSprop(self.lr), loss='mse')
        self.Q_now = model
        self.Q_next = clone_model(self.Q_now)

    def store_memory(self, s, a, r, s_):
        index = self.step_counter % self.memory_size
        self.memory[index, :] = np.hstack((s, a, r, s_))
        self.step_counter += 1

    def choose_action(self, state):
        if np.random.uniform() > self.epsilon:
            return np.argmax(self.Q_now.predict_on_batch(state.reshape(1, -1)))
        else:
            return np.random.randint(self.actions_n)

    def learn(self):
        if self.step_counter % self.update_target_iter == 0:
            self.Q_next.set_weights(self.Q_now.get_weights())
        
        batch_memory_index = np.random.choice(min(self.step_counter, self.memory_size), size=self.batch_size)
        batch_memory = self.memory[batch_memory_index, :]

        state = batch_memory[:, :self.states_n]
        action = batch_memory[:, self.states_n].astype(int)
        reward = batch_memory[:, self.states_n + 1]
        next_state = batch_memory[:, -self.states_n:]
        batch_index = np.arange(self.batch_size)
        
        Q_now = self.Q_now.predict_on_batch(state)
        Q_next = self.Q_next.predict_on_batch(next_state)
        
        y = np.copy(Q_now)
        y[batch_index, action] = reward + self.gamma * np.max(Q_next, axis=1)
        
        loss = self.Q_now.train_on_batch(state, y)

        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decrement
        else:
            self.epsilon = self.epsilon_min

        ############################
        if self.step_counter % 100000 == 0:
            '''
            print("cost: ", loss, " true_cost: ", (y - Q_now)[batch_index, action].sum(), end=' ')
            plt.figure(1, clear=True)
            plt.plot(state[:, 0], y[batch_index, action], '.', label='q_target')
            plt.plot(state[:, 0], Q_now[batch_index, action], '^', label ='q_now')
            
            Q_now = self.Q_now.predict_on_batch(state).numpy()
            print("true_cost_new: ", (y - Q_now)[batch_index, action].sum())
            plt.plot(state[:, 0], Q_now[batch_index, action], '+', label ='q_now_new')
            '''
            tmp = min(self.step_counter, self.memory_size)
            state = self.memory[:tmp, :self.states_n]
            action = self.memory[:tmp, self.states_n].astype(int)
            reward = self.memory[:tmp, self.states_n + 1]
            index = np.arange(tmp)
            Q_now = self.Q_now.predict_on_batch(state)
            plt.figure(1, clear=True)
            plt.plot(state[:, 0], reward, '.', label='reward')
            plt.plot(state[:, 0], Q_now[index, action], '+', label='q_now')
            

            plt.legend()
            plt.grid()
            plt.show()
        ############################
        return loss


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    env = env.unwrapped
    RL = DQN(env.action_space.n, env.observation_space.shape[0],
            update_target_iter=100, epsilon_decrement=0.001)

    loss= []
    pos_threshold = env.observation_space.high[0] / 2
    angle_threshold = env.observation_space.high[2] / 2

    for episode in range(100):
        state = env.reset()
        Reward = 0

        while True:
            env.render()
            action = RL.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            Reward += reward
            
            x, x_dot, theta, theta_dot = next_state
            r1 = (pos_threshold - abs(x)) / pos_threshold - 0.8
            r2 = (angle_threshold - abs(theta)) / angle_threshold - 0.5
            reward = r1 + r2

            RL.store_memory(state, action, reward, next_state)

            if RL.step_counter >= 1000:
                loss.append(RL.learn())

            if done:
                break
            state = next_state

        print("episode: {:<5}, reward: {:<10}, average reward: {:<20} in {} steps, epsilon is {}".format(episode+1, Reward, RL.step_counter / (episode+1), RL.step_counter, RL.epsilon))

    env.close()
    plt.figure(1, clear=True)
    plt.plot(np.arange(len(loss)), loss)
    plt.show()

