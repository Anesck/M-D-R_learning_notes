import gym
import numpy as np

def run_episode(env, policy=None, render=False):
    episode_reward = 0
    state = env.reset()
    while True:
        if render:
            env.render()
        if policy is None:
            action = env.action_space.sample()
        else:
            action = np.random.choice(env.action_space.n, p=policy[state])
        state, reward, done, _ = env.step(action)

        episode_reward += reward
        if done:
            break
    return episode_reward

class SARSA():
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

    def learn(self, state, action, reward, next_state, done, next_action):
        u = reward + self.gamma * self.q[next_state][next_action] * (1 - done)
        td_error = u - self.q[state][action]

if __name__ == "__main__":
    env = gym.make("Taxi-v3")

    #print(run_episode(env, render=True))

    agent = 1

    env.close()
