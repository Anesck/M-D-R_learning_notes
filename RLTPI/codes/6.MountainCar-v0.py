import gym
import numpy as np
import matplotlib.pyplot as plt

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

if __name__ =="__main__":
    env = gym.make("MountainCar-v0")

    run_episode(env, render=True)

    env.close()