import gym
import numpy as np
import matplotlib.pyplot as plt

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

class Agent():
    def __init__(self, env, gamma=0.9, learning_rate=0.1, epsilon=0.01):
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.action_n = env.action_space.n
        self.q = np.zeros((env.observation_space.n, env.action_space.n))

    def choose_action(self, state):
        if np.random.uniform() > self.epsilon:
            action = self.q[state].argmax()
        else:
            action = np.random.randint(self.action_n)
        return action

class SARSA(Agent):
    def learn(self, state, action, reward, next_state, done, next_action):
        u = reward + self.gamma * self.q[next_state][next_action] * (1 - done)
        self.q[state][action] += self.learning_rate * (u - self.q[state][action])

class ExpectedSARSA(Agent):
    def learn(self, state, action, reward, next_state, done, *args):
        v = self.q[next_state].mean() * self.epsilon + \
                self.q[next_state].max() * (1 - self.epsilon)
        u = reward + self.gamma * v * (1 - done)
        self.q[state][action] += self.learning_rate * (u - self.q[state][action])

class QLearning(Agent):
    def learn(self, state, action, reward, next_state, done, *args):
        u = reward + self.gamma * self.q[next_state].max() * (1 - done)
        self.q[state][action] += self.learning_rate * (u - self.q[state][action])

class DoubleQLearning(Agent):
    def __init__(self, env, gamma=0.9, learning_rate=0.1, epsilon=0.01):
        super().__init__(env, gamma=gamma, learning_rate=learning_rate, epsilon=epsilon)
        self.q1 = np.zeros((env.observation_space.n, env.action_space.n))

    def choose_action(self, state):
        if np.random.uniform() > self.epsilon:
            action = (self.q[state] + self.q1[state]).argmax()
        else:
            action = np.random.randint(self.action_n)
        return action

    def learn(self, state, action, reward, next_state, done, *args):
        if np.random.randint(2):
            self.q, self.q1 = self.q1, self.q
        a = self.q[next_state].argmax()
        u = reward + self.gamma * self.q1[next_state][a] * (1 - done)
        self.q[state][action] += self.learning_rate * (u - self.q[state][action])

class SARSALambda(Agent):
    def __init__(self, env, lamb=0.5, beta=1, gamma=0.9, learning_rate=0.1, epsilon=0.01):
        super().__init__(env, gamma=gamma, learning_rate=learning_rate, epsilon=epsilon)
        self.lamb = lamb
        self.beta = beta
        self.e = np.zeros((env.observation_space.n, env.action_space.n))

    def learn(self, state, action, reward, next_state, done, next_action):
        self.e *= (self.lamb * self.gamma)
        self.e[state][action] = 1 + self.beta * self.e[state][action]
        u = reward + self.gamma * self.q[next_state][next_action] * (1 - done)
        self.q += self.learning_rate * self.e * (u - self.q[state][action])
        # self.q += self.learning_rate * self.e * (u - self.q)
        if done:    # 为下一回合初始化资格迹
            self.e *= 0

if __name__ == "__main__":
    env = gym.make("Taxi-v3")

    #print(run_episode(env, render=True))

    episodes = 5000
    episode_rewards = []

    agent = SARSA(env)
    #agent = ExpectedSARSA(env)
    #agent = QLearning(env)
    #agent = DoubleQLearning(env)
    #agent = SARSALambda(env)
    
    # 智能体的训练与测试
    print("训练智能体中")
    for i in tqdm(range(episodes)):
        episode_rewards.append(run_episode(env, agent, train=True))
    plt.plot(episode_rewards)
    agent.epsilon = 0
    print("测试智能体中")
    episode_rewards = [run_episode(env, agent) for _ in tqdm(range(100))]
    print("平均回合奖励 = {} / {} = {}".format(sum(episode_rewards), \
            len(episode_rewards), np.mean(episode_rewards)))
    plt.show()

    env.close()
