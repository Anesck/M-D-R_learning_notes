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

class QAC():
    def __init__(self, env, actor_kwargs, critic_kwargs, gamma=0.99):
        self.action_n = env.action_space.n
        self.gamma = gamma
        self.discount = 1
        self.actor_net = self.build_network(output_size=self.action_n, \
                output_activation=tf.nn.softmax, \
                loss=keras.losses.categorical_crossentropy, **actor_kwargs)
        self.critic_net = self.build_network(output_size=self.action_n, **critic_kwargs)

    def build_network(self, hidden_sizes, output_size, input_size=None, \
            activation=tf.nn.relu, output_activation=None, \
            loss=keras.losses.mse, learning_rate=0.01):
        model = keras.Sequential()
        for idx, hidden_sizes in enumerate(hidden_sizes):
            kwargs = {}
            if idx == 0 and input_size is not None:
                kwargs["input_shape"] = (input_size, )
            model.add(keras.layers.Dense(units=hidden_sizes, activation=activation, \
                    kernel_initializer=keras.initializers.GlorotUniform(seed=0), **kwargs))
        model.add(keras.layers.Dense(units=output_size, activation=output_activation, \
                kernel_initializer=keras.initializers.GlorotUniform(seed=0)))
        model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss=loss)
        return model

    def choose_action(self, state):
        probs = self.actor_net.predict(state[np.newaxis])[0]
        return np.random.choice(self.action_n, p=probs)

    def learn(self, state, action, reward, next_state, done, next_action=None):
        x = state[np.newaxis]
        u = self.critic_net.predict(x)
        q = u[0, action]
        x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
        with tf.GradientTape() as tape:
            pi_tensor = self.actor_net(x_tensor)[0, action]
            logpi_tensor = tf.math.log(tf.clip_by_value(pi_tensor, 1e-6, 1.))
            loss_tensor = -self.discount * q * logpi_tensor
        grad_tensors = tape.gradient(loss_tensor, self.actor_net.variables)
        self.actor_net.optimizer.apply_gradients(zip(grad_tensors, self.actor_net.variables))

        u[0, action] = reward
        if not done:
            q = self.critic_net.predict(next_state[np.newaxis])[0, next_action]
            u[0, action] += self.gamma * q
        self.critic_net.fit(x, u, verbose=0)

        if done:
            self.discount = 1
        else:
            self.discount *= self.gamma

class AdvantageAC(QAC):
    def __init__(self, env, actor_kwargs, critic_kwargs, gamma=0.99):
        super().__init__(env=env, actor_kwargs=actor_kwargs, \
                critic_kwargs=critic_kwargs, gamma=gamma)
        self.critic_net = self.build_network(output_size=1, **critic_kwargs)

    def learn(self, state, action, reward, next_state, done, *args):
        x = state[np.newaxis]
        u = reward + self.gamma * self.critic_net.predict( \
                next_state[np.newaxis]) * (1 - done)
        td_error = u - self.critic_net.predict(x)

        x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
        with tf.GradientTape() as tape:
            pi_tensor = self.actor_net(x_tensor)[0, action]
            logpi_tensor = tf.math.log(tf.clip_by_value(pi_tensor, 1e-6, 1.))
            loss_tensor = -self.discount * td_error * logpi_tensor
        grad_tensors = tape.gradient(loss_tensor, self.actor_net.variables)
        self.actor_net.optimizer.apply_gradients(zip(grad_tensors, self.actor_net.variables))

        self.critic_net.fit(x, u, verbose=0)

        if done:
            self.discount = 1
        else:
            self.discount *= self.gamma

class ElibilityTraceAC(QAC):
    def __init__(self, env, actor_kwargs, critic_kwargs, gamma=0.99, \
            actor_lambda=0.9, critic_lambda=0.9):
        super().__init__(env=env, actor_kwargs=actor_kwargs, \
                critic_kwargs=critic_kwargs, gamma=gamma)
        self.actor_lambda = actor_lambda
        self.critic_lambda = critic_lambda
        state_dim = env.observation_space.shape[0]
        self.actor_net = self.build_network(input_size=state_dim, \
                output_size=self.action_n, output_activation=tf.nn.softmax, **actor_kwargs)
        self.critic_net = self.build_network(input_size=state_dim, output_size=1, **critic_kwargs)
        self.actor_traces = [np.zeros_like(weight) for weight in self.actor_net.get_weights()]
        self.critic_traces = [np.zeros_like(weight) for weight in self.critic_net.get_weights()]

    def learn(self, state, action, reward, next_state, done, *args):
        v = self.critic_net.predict(state[np.newaxis])[0, 0]
        u = reward + self.gamma * self.critic_net.predict( \
                next_state[np.newaxis])[0, 0] * (1 - done)
        td_error = u - v

        x_tensor = tf.convert_to_tensor(state[np.newaxis], dtype=tf.float32)
        with tf.GradientTape() as tape:
            pi_tensor = self.actor_net(x_tensor)
            logpi_tensor = tf.math.log(tf.clip_by_value(pi_tensor, 1e-6, 1.))
            logpi_pick_tensor = logpi_tensor[0, action]
        grad_tensors = tape.gradient(logpi_pick_tensor, self.actor_net.variables)
        self.actor_traces = [self.gamma * self.actor_lambda * trace + self.discount * \
                grad.numpy() for trace, grad in zip(self.actor_traces, grad_tensors)]
        actor_grads = [tf.convert_to_tensor(-td_error * trace, \
                dtype=tf.float32) for trace in self.actor_traces]
        actor_grads_and_vars = tuple(zip(actor_grads, self.actor_net.variables))
        self.actor_net.optimizer.apply_gradients(actor_grads_and_vars)

        with tf.GradientTape() as tape:
            v_tensor = self.critic_net(x_tensor)
        grad_tensors = tape.gradient(v_tensor, self.critic_net.variables)
        self.critic_traces = [self.gamma * self.critic_lambda * trace + grad.numpy() \
                for trace, grad in zip(self.critic_traces, grad_tensors)]
        critic_grads = [tf.convert_to_tensor(-td_error * trace, \
                dtype=tf.float32) for trace in self.critic_traces]
        critic_grads_and_vars = tuple(zip(critic_grads, self.critic_net.variables))
        self.critic_net.optimizer.apply_gradients(critic_grads_and_vars)

        if done:
            self.actor_traces = [np.zeros_like(weight) \
                    for weight in self.actor_net.get_weights()]
            self.critic_traces = [np.zeros_like(weight) \
                    for weight in self.critic_net.get_weights()]
            self.discount = 1
        else:
            self.discount *= self.gamma

if __name__ == "__main__":
    env = gym.make("Acrobot-v1")
    #env = env.unwrapped
    
    episodes = 100
    episode_rewards = []

    #run_episode(env, render=True)
    episode_rewards = [run_episode(env) for _ in tqdm(range(100))]
    print("随机策略的平均回合奖励 = {} / {} = {}".format(sum(episode_rewards), \
            len(episode_rewards), np.mean(episode_rewards)))

    actor_kwargs = {"hidden_sizes": [100, ], "learning_rate" : 0.0002}
    critic_kwargs = {"hidden_sizes": [100, ], "learning_rate" : 0.0005}

    agent = QAC(env, actor_kwargs=actor_kwargs, critic_kwargs=critic_kwargs)
    #agent = AdvantageAC(env, actor_kwargs=actor_kwargs, critic_kwargs=critic_kwargs)
    #agent = ElibilityTraceAC(env, actor_kwargs=actor_kwargs, critic_kwargs=critic_kwargs)

    # 智能体的训练与测试
    print("训练智能体中...")
    for i in tqdm(range(episodes)):
        episode_rewards.append(run_episode(env, agent, train=True))
    plt.plot(episode_rewards)
    print("测试智能体中...")
    episode_rewards = [run_episode(env, agent) for _ in tqdm(range(100))]
    print("平均回合奖励 = {} / {} = {}".format(sum(episode_rewards), \
            len(episode_rewards), np.mean(episode_rewards)))
    plt.show()

    env.close()