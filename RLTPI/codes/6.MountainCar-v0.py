import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras

from tqdm import tqdm

def run_episode(env, agent=None, train=False, render=False):
    episode_reward = 0
    state = env.reset()
    if agent is None:
        action = env.action_space.sample()
    else:
        action = agent.choose_action(state)
    while True:
        if render:
            env.render()
        next_state, reward, done, _ = env.step(action)
        if agent is None:
            next_action = env.action_space.sample()
        else:
            next_action = agent.choose_action(next_state)
            if train:
                agent.learn(state, action, reward, next_state, done, next_action)

        episode_reward += reward
        if done:
            break
        state, action = next_state, next_action
    return episode_reward

class TileCoder():
    def __init__(self, layers, features):
        self.layers = layers
        self.features = features
        self.codebook = {}

    def get_feature(self, codeword):
        if codeword in self.codebook:
            return self.codebook[codeword]
        count = len(self.codebook)
        if count >= self.features:
            return hash(codeword) % self.features
        else:
            self.codebook[codeword] = count
            return count

    def __call__(self, floats=(), ints=()):
        dim = len(floats)
        scaled_floats = tuple(f * self.layers * self.layers for f in floats)
        features = []
        for layer in range(self.layers):
            codeword = (layer, ) + tuple(int((f + (1 + dim * i) * layer) \
                    / self.layers) for i, f in enumerate(scaled_floats)) + ints
            feature = self.get_feature(codeword)
            features.append(feature)
        return features

class SARSA():
    def __init__(self, env, layers=8, features=1893, gamma=1., \
            learning_rate=0.03, epsilon=0.001):
        self.action_n = env.action_space.n
        self.obs_low = env.observation_space.low
        self.obs_scale = env.observation_space.high - env.observation_space.low
        self.encoder = TileCoder(layers, features)
        self.w = np.zeros(features)
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon

    def encode(self, state, action):
        states = tuple((state - self.obs_low) / self.obs_scale)
        return self.encoder(states, (action,))

    def get_q(self, state, action):
        features = self.encode(state, action)
        return self.w[features].sum()

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_n)
        qs = [self.get_q(state, action) for action in range(self.action_n)]
        return np.argmax(qs)

    def learn(self, state, action, reward, next_state, done, next_action):
        u = reward + self.gamma * self.get_q(next_state, next_action) * (1 - done)
        td_error = u - self.get_q(state, action)
        features = self.encode(state, action)
        self.w[features] += (self.learning_rate * td_error)

class SARSALambda(SARSA):
    def __init__(self, env, layers=8, features=1893, gamma=1., \
            learning_rate=0.03, epsilon=0.001, lamb=0.9):
        super().__init__(env=env, layers=layers, features=features, \
                gamma=gamma, learning_rate=learning_rate, epsilon=epsilon)
        self.lamb = lamb
        self.z = np.zeros(features)

    def learn(self, state, action, reward, next_state, done, next_action):
        u = reward
        if not done:
            u += (self.gamma * self.get_q(next_state, next_action))
            self.z *= self.gamma * self.lamb
            features = self.encode(state, action)
            self.z[features] = 1.
        td_error = u - self.get_q(state, action)
        self.w += (self.learning_rate * td_error * self.z)
        if done:
            self.z = np.zeros_like(self.z)

class Replayer():
    def __init__(self, capacity):
        self.memory = pd.DataFrame(index=range(capacity), \
                columns=["state", "action", "reward", "next_state", "done"])
        self.i = 0
        self.count = 0
        self.capacity = capacity

    def store_memory(self, *args):
        self.memory.loc[self.i] = args
        self.i = (self.i + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def sample(self, size):
        indices = np.random.choice(self.count, size=size)
        return (np.stack(self.memory.loc[indices, field]) \
                for field in self.memory.columns)

class DQN():
    def __init__(self, env, net_kwargs={}, gamma=0.99, epsilon=0.001, \
            replayer_capacity=10000, batch_size=64):
        self.action_n = env.action_space.n
        self.gamma = gamma
        self.epsilon = epsilon

        self.batch_size = batch_size
        self.replayer = Replayer(replayer_capacity)

        state_dim = env.observation_space.shape[0]
        self.evaluate_net = self.build_network(input_size=state_dim, \
                output_size=self.action_n, **net_kwargs)
        self.target_net = keras.models.clone_model(self.evaluate_net)
    
    def build_network(self, input_size, hidden_sizes, output_size, \
            activation=tf.nn.relu, output_activation=None, learning_rate=0.01):
        model = keras.Sequential()
        for layer, hidden_size in enumerate(hidden_sizes):
            kwargs = dict(input_shape=(input_size, )) if not layer else {}
            model.add(keras.layers.Dense(units=hidden_size, activation=activation, \
                    kernel_initializer=keras.initializers.GlorotUniform(seed=0), **kwargs))
        model.add(keras.layers.Dense(units=output_size, activation=output_activation, \
                kernel_initializer=keras.initializers.GlorotUniform(seed=0)))
        model.compile(loss="mse", optimizer=keras.optimizers.Adam(lr=learning_rate))
        return model

    def learn(self, state, action, reward, next_state, done, *args):
        self.replayer.store_memory(state, action, reward, next_state, done)
        states, actions, rewards, next_states, dones = \
                self.replayer.sample(self.batch_size)

        next_qs = self.target_net.predict(next_states)
        us = rewards + self.gamma * next_qs.max(axis=-1) * (1 - dones)
        targets = self.evaluate_net.predict(states)
        targets[np.arange(us.shape[0]), actions] = us
        self.evaluate_net.fit(states, targets, verbose=0)

        if done:
            self.target_net.set_weights(self.evaluate_net.get_weights())

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_n)
        qs = self.evaluate_net.predict(state[np.newaxis])
        return np.argmax(qs)

class DoubleDQN(DQN):
    def learn(self, state, action, reward, next_state, done, *args):
        self.replayer.store_memory(state, action, reward, next_state, done)
        states, actions, rewards, next_states, dones = \
                self.replayer.sample(self.batch_size)

        next_eval_qs = self.evaluate_net.predict(next_states)
        next_actions = next_eval_qs.argmax(axis=-1)
        next_qs = self.target_net.predict(next_states)
        next_max_qs = next_qs[np.arange(next_qs.shape[0]), next_actions]
        us = rewards + self.gamma * next_max_qs * (1 - dones)
        targets = self.evaluate_net.predict(states)
        targets[np.arange(us.shape[0]), actions] = us
        self.evaluate_net.fit(states, targets, verbose=0)

        if done:
            self.target_net.set_weights(self.evaluate_net.get_weights())

if __name__ =="__main__":
    env = gym.make("MountainCar-v0")
    env = env.unwrapped
    print("观测空间 = {}".format(env.observation_space))
    print("动作空间 = {}".format(env.action_space))
    print("位置范围 = {}".format((env.min_position, env.max_position)))
    print("速度范围 = {}".format((-env.max_speed, env.max_speed)))
    print("目标位置 = {}".format(env.goal_position))

    #run_episode(env, render=True)
    
    episodes = 100
    episode_rewards = []

    agent = SARSA(env)
    #agent = SARSALambda(env)
    #agent = DQN(env, net_kwargs={"hidden_sizes": [64, ], "learning_rate": 0.001})
    #agent = DoubleDQN(env, net_kwargs={"hidden_sizes": [64, ], "learning_rate": 0.01})

    # 智能体的训练与测试
    print("训练智能体中...")
    for i in tqdm(range(episodes)):
        episode_rewards.append(run_episode(env, agent, train=True))
    plt.plot(episode_rewards)
    agent.epsilon = 0
    print("测试智能体中...")
    episode_rewards = [run_episode(env, agent) for _ in tqdm(range(100))]
    print("平均回合奖励 = {} / {} = {}".format(sum(episode_rewards), \
            len(episode_rewards), np.mean(episode_rewards)))
    plt.show()

    env.close()
