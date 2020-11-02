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
        if agent is None or agent.random_behavior:
            action = env.action_space.sample()
        else:
            action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)

        episode_reward += reward
        if train:
            agent.learn(state, action, reward, done)
        if done:
            break
        state = next_state
    return episode_reward

class VPG():
    def __init__(self, env, policy_kwargs, baseline_kwargs=None, gamma=0.99, offpolicy=False):
        self.action_n = env.action_space.n
        self.gamma= gamma
        self.trajectory = []

        if not offpolicy:
            self.random_behavior = False
            policy_loss = keras.losses.categorical_crossentropy
        else:
            self.random_behavior = True
            policy_loss = lambda y_true, y_pred: -tf.reduce_sum(y_true * y_pred, axis=-1)

        self.policy_net = self.build_network(output_size=self.action_n, \
                output_activation=tf.nn.softmax, loss=policy_loss, **policy_kwargs)
        if baseline_kwargs:
            self.baseline_net = self.build_network(output_size=1, **baseline_kwargs)

    def build_network(self, hidden_sizes, output_size, activation=tf.nn.relu, \
            output_activation=None, loss=keras.losses.mse, learning_rate=0.01):
        model = keras.Sequential()
        for hidden_size in hidden_sizes:
            model.add(keras.layers.Dense(units=hidden_size, activation=activation))
        model.add(keras.layers.Dense(units=output_size, activation=output_activation))
        model.compile(loss=loss, optimizer=keras.optimizers.Adam(lr=learning_rate))
        return model

    def choose_action(self, state):
        probs = self.policy_net.predict(state[np.newaxis])[0]
        return np.random.choice(self.action_n, p=probs)

    def learn(self, state, action, reward, done):
        self.trajectory.append((state, action, reward))

        if done:
            df = pd.DataFrame(self.trajectory, columns=["state", "action", "reward"])
            df["discount"] = self.gamma ** df.index.to_series()
            df["psi"] = (df["discount"] * df["reward"])[::-1].cumsum()

            x = np.stack(df["state"])
            if hasattr(self, "baseline_net"):
                df["return"] = df["psi"] / df["discount"]
                y = df["return"].values[:, np.newaxis]
                self.baseline_net.train_on_batch(x, y)
                df["baseline"] = self.baseline_net.predict(x)
                df["psi"] -= (df["discount"] * df["baseline"])
            if self.random_behavior:
                df["psi"] *= self.action_n

            df["psi"] = (df["psi"] - df["psi"].mean()) / df["psi"].std()
            sample_weight = df["psi"].values[:, np.newaxis]
            y = np.eye(self.action_n)[df["action"]]
            self.policy_net.train_on_batch(x, y, sample_weight=sample_weight)
            self.trajectory = []

if __name__ =="__main__":
    env = gym.make("CartPole-v0")
    #env = env.unwrapped
    
    episodes = 100
    episode_rewards = []

    #run_episode(env, render=True)
    episode_rewards = [run_episode(env) for _ in tqdm(range(100))]
    print("随机策略的平均回合奖励 = {} / {} = {}".format(sum(episode_rewards), \
            len(episode_rewards), np.mean(episode_rewards)))

    policy_kwargs = {"hidden_sizes": [10, ], "activation": tf.nn.relu, "learning_rate" : 0.1}
    baseline_kwargs = {"hidden_sizes": [10, ], "activation": tf.nn.relu, "learning_rate" : 0.1}

    agent = VPG(env, policy_kwargs=policy_kwargs)
    #agent = VPG(env, policy_kwargs=policy_kwargs, baseline_kwargs=baseline_kwargs)
    #agent = VPG(env, policy_kwargs=policy_kwargs, offpolicy=True)
    #agent = VPG(env, policy_kwargs=policy_kwargs, baseline_kwargs=baseline_kwargs, offpolicy=True)

    # 智能体的训练与测试
    print("训练智能体中...")
    for i in tqdm(range(episodes)):
        episode_rewards.append(run_episode(env, agent, train=True))
    plt.plot(episode_rewards)
    agent.random_behavior = False
    print("测试智能体中...")
    episode_rewards = [run_episode(env, agent) for _ in tqdm(range(100))]
    print("平均回合奖励 = {} / {} = {}".format(sum(episode_rewards), \
            len(episode_rewards), np.mean(episode_rewards)))
    plt.show()

    env.close()
