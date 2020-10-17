import gym
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['simhei']

def run_episode(env, policy=None, render=False):
    def sum_hand(hand):
        if 1 in hand and sum(hand) + 10 <= 21:
            return sum(hand) + 10
        return sum(hand)

    def render_text():
        print("真实环境： (玩家, 庄家)点数 / 手牌 = \t({}, {}) / ({}, {})".format(sum_hand(env.player), sum_hand(env.dealer), env.player, env.dealer))
        print("观测环境： (玩家, 庄家, 玩家A为11) = \t{}\t\t玩家要牌 = {}\n".format(state, bool(action)))

    episode_reward = 0
    state = env.reset()
    while True:
        if policy is None:
            action = env.action_space.sample()
        else:
            state = (state[0], state[1], int(state[2]))
            action = np.random.choice(env.action_space.n, p=policy[state])
        if render:
            render_text()
        next_state, reward, done, _ = env.step(action)
        state = next_state

        episode_reward += reward
        if done:
            if render:
                render_text()
                print("回合奖励值：{}".format(episode_reward))
            break
    return episode_reward

def test_policy(env, policy, episode_num=50000):
    win, tied, lost = 0, 0, 0
    for _ in range(episode_num):
        score = run_episode(env, policy)
        if score == 1:
            win += 1
        elif score == 0:
            tied += 1
        elif score == -1:
            lost += 1
    print("AI win {} round, tied {} round, lost {} round, in {} round, against dealer."\
            .format(win, tied, lost, episode_num))

def plot(data, suptitle="策略评估得到的状态函数（颜色越浅表示状态价值越大）"):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    titles = ["without ace", "with ace"]
    have_aces = [0, 1]
    extent = [12, 22, 1, 11]
    for title, have_ace, axis in zip(titles, have_aces, axes):
        dat = data[extent[0]:extent[1], extent[2]:extent[3], have_ace].T
        axis.imshow(dat, extent=extent, origin="upper")
        axis.set_xlabel("player sum")
        axis.set_ylabel("dealer showing")
        axis.set_title(title)
    plt.suptitle(suptitle)
    plt.show()

def explore_sampling(env, policy):
    state_actions = []
    state = env.reset()
    while True:
        state = (state[0], state[1], int(state[2]))
        action = np.random.choice(env.action_space.n, p=policy[state])
        state_actions.append((state, action))
        state, reward, done, _ = env.step(action)
        if done:
            break
    return state_actions, reward

def evaluate_action_monte_carlo(env, policy, episode_num=500000):
    q = np.zeros_like(policy)
    c = np.zeros_like(policy)
    for _ in range(episode_num):
        state_actions, reward = explore_sampling(env, policy)
        g = reward
        for state, action in state_actions:
            c[state][action] += 1.
            q[state][action] += (g - q[state][action]) / c[state][action]
    return q

def evaluate_monte_carlo_importance_sample(env, policy, behavior_policy, episode_num=500000):
    q = np.zeros_like(policy)
    c = np.zeros_like(policy)
    for _ in range(episode_num):
        state_actions, reward = explore_sampling(env, behavior_policy)
        g = reward
        rho = 1.
        for state, action in reversed(state_actions):
            c[state][action] += rho
            q[state][action] += (rho / c[state][action] * (g - q[state][action]))
            rho *= (policy[state][action] / behavior_policy[state][action])
            if rho == 0:
                break
    return q

def monte_carlo_with_exploring_start(env, episode_num=500000):
    policy = np.zeros((22, 11, 2, 2))
    policy[:, :, :, 1] = 1.
    q = np.zeros_like(policy)
    c = np.zeros_like(policy)
    for _ in range(episode_num):
        state = (np.random.randint(12, 22), \
                 np.random.randint(1, 11), \
                 np.random.randint(2))
        action = np.random.randint(2)
        env.reset()
        if state[2]:
            env.player = [1, state[0] - 11]
        else:
            if state[0] == 21:
                env.player = [10, 9, 2]
            else:
                env.player = [10, state[0] - 10]
        env.dealer[0] = state[1]
        state_actions = []
        while True:
            state_actions.append((state, action))
            state, reward, done, _ = env.step(action)
            if done:
                break
            state = (state[0], state[1], int(state[2]))
            action = np.random.choice(env.action_space.n, p=policy[state])

        g = reward
        for state, action in state_actions:
            c[state][action] += 1.
            q[state][action] += (g - q[state][action]) / c[state][action]
            a = q[state].argmax()
            policy[state] = 0.
            policy[state][a] = 1.
    return policy, q

def monte_carlo_with_soft(env, episode_num=500000, epsilon=0.1):
    policy = np.ones((22, 11, 2, 2)) * 0.5
    q = np.zeros_like(policy)
    c = np.zeros_like(policy)
    for _ in range(episode_num):
        state_actions, reward = explore_sampling(env, policy)
        g = reward
        for state, action in state_actions:
            c[state][action] += 1.
            q[state][action] += (g - q[state][action]) / c[state][action]
            a = q[state].argmax()
            policy[state] = epsilon / env.action_space.n
            policy[state][a] += (1. - epsilon)
    return policy, q

def monte_carlo_importance_sample(env, episode_num=500000):
    policy = np.zeros((22, 11, 2, 2))
    policy[:, :, : 0] = 1
    behavior_policy = np.ones_like(policy) * 0.5
    q = np.zeros_like(policy)
    c = np.zeros_like(policy)
    for _ in range(episode_num):
        state_actions, reward = explore_sampling(env, behavior_policy)
        g = reward
        rho = 1.
        for state, action in reversed(state_actions):
            c[state][action] += rho
            q[state][action] += (rho / c[state][action] * (g - q[state][action]))
            a = q[state].argmax()
            policy[state] = 0.
            policy[state][a] = 1.
            if a != action:
                break
            rho /= behavior_policy[state][action]
    return policy, q

if __name__ == "__main__":
    env = gym.make("Blackjack-v0")
    np.set_printoptions(precision=3, suppress=True)

    #run_episode(env, render=True)

    # 目标策略和行为策略
    policy = np.zeros((22, 11, 2, 2))
    policy[20:, :, :, 0] = 1    # 手牌 >= 20 时不再要牌
    policy[:20, :, :, 1] = 1    # 手牌 < 20 时继续要牌
    behavior_policy = np.ones_like(policy) * 0.5
    #test_policy(env, policy)
    #test_policy(env, behavior_policy)

    # 策略评估
    '''
    q = evaluate_action_monte_carlo(env, policy)   # 同策回合更新
    #q = evaluate_monte_carlo_importance_sample(env, policy, behavior_policy)    # 重要性采样
    v = (q * policy).sum(axis=-1)
    '''
    
    # 最优策略求解
    '''
    policy, q = monte_carlo_with_exploring_start(env)  # 带起始探索的同策回合更新
    #policy, q = monte_carlo_with_soft(env)  # 基于柔性策略的同策回合更新
    #policy, q = monte_carlo_importance_sample(env)
    v = q.max(axis=-1)
    test_policy(env, policy)
    plot(policy.argmax(-1), suptitle="最优策略（黄色表示继续要牌，紫色表示不要）")
    '''

    #plot(v)

    env.close()
