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
    while True:
        if render:
            env.render()
        if agent is None:
            action = env.action_space.sample()
        else:
            action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        
        episode_reward += reward
        if train:
            agent.learn(state, action, reward, next_state, done)
        if done:
            break
        state = next_state
    return episode_reward

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

class OrnsteinUhlenbeckProcess():
    def __init__(self, size, mu=0, sigma=1, theta=0.15, dt=0.01):
        self.size = size
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt

    def reset(self, x=0.):
        self.x = x * np.ones(self.size)

    def __call__(self):
        n = np.random.normal(size=self.size)
        self.x += (self.theta * (self.mu - self.x) * self.dt + \
                self.sigma * np.sqrt(self.dt) * n)
        return self.x

class DDPG():
    def __init__(self, env, actor_kwargs, critic_kwargs, \
            replayer_capacity=100000, replayer_initial_transitions=10000, \
            gamma=0.99, batches=1, batch_size=64, \
            net_learning_rate=0.005, noise_scale=0.1, explore=True):
        self.action_low = env.action_space.low
        self.action_high = env.action_space.high
        self.gamma = gamma
        self.net_learning_rate = net_learning_rate
        self.explore = explore

        self.batches = batches
        self.batch_size = batch_size
        self.replayer = Replayer(replayer_capacity)
        self.replayer_initial_transitions = replayer_initial_transitions

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        state_action_dim = state_dim + action_dim
        self.noise = OrnsteinUhlenbeckProcess(size=(action_dim, ), sigma=noise_scale)
        self.noise.reset()

        self.actor_evaluate_net = self.build_network(input_size=state_dim, **actor_kwargs)
        self.critic_evaluate_net = self.build_network(input_size=state_action_dim, **critic_kwargs)
        self.actor_target_net = keras.models.clone_model(self.actor_evaluate_net)
        self.critic_target_net = keras.models.clone_model(self.critic_evaluate_net)
        self.update_target_net(self.actor_target_net, self.actor_evaluate_net)
        self.update_target_net(self.critic_target_net, self.critic_evaluate_net)

    def update_target_net(self, target_net, evaluate_net, learning_rate=1.):
        target_weights = target_net.get_weights()
        evaluate_weights = evaluate_net.get_weights()
        average_weights = [(1 - learning_rate) * t + learning_rate * e \
                for t, e in zip(target_weights, evaluate_weights)]
        target_net.set_weights(average_weights)

    def build_network(self, input_size, hidden_sizes, output_size=1, \
            activation=tf.nn.relu, output_activation=None, \
            loss=keras.losses.mse, learning_rate=None):
        model = keras.Sequential()
        for layer, hidden_size in enumerate(hidden_sizes):
            kwargs = {"input_shape": (input_size, )} if not layer else {}
            model.add(keras.layers.Dense(units=hidden_size, activation=activation, \
                    kernel_initializer=keras.initializers.GlorotUniform(seed=0), **kwargs))
        model.add(keras.layers.Dense(units=output_size, activation=output_activation, \
                kernel_initializer=keras.initializers.GlorotUniform(seed=0)))
        model.compile(loss=loss, optimizer=keras.optimizers.Adam(learning_rate))
        return model

    def choose_action(self, state):
        if self.explore and self.replayer.count < self.replayer_initial_transitions:
            return np.random.uniform(self.action_low, self.action_high)

        action = self.actor_evaluate_net.predict(state[np.newaxis])[0]
        if self.explore:
            noise = self.noise()
            action = np.clip(action + noise, self.action_low, self.action_high)
        return action

    def learn(self, state, action, reward, next_state, done):
        self.replayer.store_memory(state, action, reward, next_state, done)

        if self.replayer.count >= self.replayer_initial_transitions:
            if done:
                self.noise.reset()

            for batch in range(self.batches):
                states, actions, rewards, next_states, dones = \
                        self.replayer.sample(self.batch_size)

                state_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
                with tf.GradientTape() as tape:
                    action_tensor = self.actor_evaluate_net(state_tensor)
                    input_tensor = tf.concat([state_tensor, action_tensor], axis=1)
                    q_tensor = self.critic_evaluate_net(input_tensor)
                    loss_tensor = -tf.reduce_mean(q_tensor)
                grad_tensors = tape.gradient(loss_tensor, self.actor_evaluate_net.variables)
                self.actor_evaluate_net.optimizer.apply_gradients(zip(grad_tensors, \
                            self.actor_evaluate_net.variables))

                next_actions = self.actor_target_net.predict(next_states)
                state_actions = np.hstack([states, actions])
                next_state_actions = np.hstack([next_states, next_actions])
                next_qs = self.critic_target_net.predict(next_state_actions)[:, 0]
                targets = rewards + self.gamma * next_qs * (1 - dones)
                self.critic_evaluate_net.fit(state_actions, targets, verbose=0)

                self.update_target_net(self.actor_target_net, \
                        self.actor_evaluate_net, self.net_learning_rate)
                self.update_target_net(self.critic_target_net, \
                        self.critic_evaluate_net, self.net_learning_rate)

class TD3(DDPG):
    def __init__(self, env, actor_kwargs, critic_kwargs, \
    		replayer_capacity=1000000, replayer_initial_transitions=10000, \
    		gamma=0.99, batches=1, batch_size=64, \
    		net_learning_rate=0.005, noise_scale=0.1, explore=True):
        super().__init__(env, actor_kwargs, critic_kwargs, replayer_capacity, \
        		replayer_initial_transitions, gamma, batches, batch_size, \
        		net_learning_rate, noise_scale, explore)

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        state_action_dim = state_dim + action_dim

        self.critic1_evaluate_net = self.build_network(input_size=state_action_dim, **critic_kwargs)
        self.critic1_target_net = keras.models.clone_model(self.critic1_evaluate_net)
        self.update_target_net(self.critic1_target_net, self.critic1_evaluate_net)

    def learn(self, state, action, reward, next_state, done):
        self.replayer.store_memory(state, action, reward, next_state, done)

        if self.replayer.count >= self.replayer_initial_transitions:
            if done:
                self.noise.reset()

            for batch in range(self.batches):
                states, actions, rewards, next_states, dones = \
                        self.replayer.sample(self.batch_size)

                state_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
                with tf.GradientTape() as tape:
                    action_tensor = self.actor_evaluate_net(state_tensor)
                    input_tensor = tf.concat([state_tensor, action_tensor], axis=1)
                    q_tensor = self.critic_evaluate_net(input_tensor)
                    loss_tensor = -tf.reduce_mean(q_tensor)
                grad_tensors = tape.gradient(loss_tensor, self.actor_evaluate_net.variables)
                self.actor_evaluate_net.optimizer.apply_gradients(zip(grad_tensors, \
                            self.actor_evaluate_net.variables))

                next_actions = self.actor_target_net.predict(next_states)
                state_actions = np.hstack([states, actions])
                next_state_actions = np.hstack([next_states, next_actions])
                next_q0s = self.critic_target_net.predict(next_state_actions)[:, 0]
                next_q1s = self.critic1_target_net.predict(next_state_actions)[:, 0]
                next_qs = np.minimum(next_q0s, next_q1s)
                targets = rewards + self.gamma * next_qs * (1 - dones)
                self.critic_evaluate_net.fit(state_actions, targets[:, np.newaxis], verbose=0)
                self.critic1_evaluate_net.fit(state_actions, targets[:, np.newaxis], verbose=0)

                self.update_target_net(self.actor_target_net, \
                        self.actor_evaluate_net, self.net_learning_rate)
                self.update_target_net(self.critic_target_net, \
                        self.critic_evaluate_net, self.net_learning_rate)
                self.update_target_net(self.critic1_target_net, \
                        self.critic1_evaluate_net, self.net_learning_rate)

if __name__ == "__main__":
    env = gym.make("Pendulum-v0")
    #env = env.unwrapped
    
    episodes = 100
    episode_rewards = []

    #run_episode(env, render=True)
    episode_rewards = [run_episode(env) for _ in tqdm(range(100))]
    print("随机策略的平均回合奖励 = {} / {} = {}".format(sum(episode_rewards), \
            len(episode_rewards), np.mean(episode_rewards)))

    actor_kwargs = {"hidden_sizes": [32, 64], "learning_rate": 0.0001}
    critic_kwargs = {"hidden_sizes": [64, 128], "learning_rate": 0.001}

    #agent = DDPG(env, actor_kwargs=actor_kwargs, critic_kwargs=critic_kwargs)
    agent = TD3(env, actor_kwargs=actor_kwargs, critic_kwargs=critic_kwargs)

    # 智能体的训练与测试
    print("训练智能体中...")
    for i in tqdm(range(episodes)):
        episode_rewards.append(run_episode(env, agent, train=True))
    plt.plot(episode_rewards)
    print("测试智能体中...")
    agent.explore = False
    episode_rewards = [run_episode(env, agent, render=True) for _ in tqdm(range(100))]
    print("平均回合奖励 = {} / {} = {}".format(sum(episode_rewards), \
            len(episode_rewards), np.mean(episode_rewards)))
    plt.show()

    env.close()
