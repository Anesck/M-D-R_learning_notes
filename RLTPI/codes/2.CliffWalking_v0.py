import gym
import scipy
import numpy as np

def get_env_info(env):
    print('observation space = {}'.format(env.observation_space))
    print('action space = {}'.format(env.action_space))
    print('number of states = {}'.format(env.nS))
    print('number of actions = {}'.format(env.nA))
    print('size of map = {}'.format(env.shape))

# if policy is None, use input data to test and explore environment
def run_episode(env, policy=None):
    total_reward = 0
    state = env.reset()
    while True:
        if policy is None:
            print("state = {}".format(state))
            env.render()
            action = int(input("input action:"))
        else:
            action = np.random.choice(env.nA, p=policy[state])
            env.render()
        state, reward, done, _ = env.step(action)
        total_reward += reward

        if policy is None:
            print("reward = {}".format(reward))
        if done:
            break
    return total_reward

# optimal policy of artificial selection
def get_artificial_optimal_policy(env):
    actions = np.ones(env.shape, dtype=int)
    actions[-1, :] = 0
    actions[:, -1] = 2
    optimal_policy = np.eye(env.nA)[actions.reshape(-1, )]   # convert to OneHotkey represent probability
    return optimal_policy

def disp_policy_as_map(env, policy):
    actions = ['up', 'right', 'down', 'left']
    map_row, map_column = env.shape
    for row in range(map_row):
        for column in range(map_column):
            state = row * map_column + column
            print("{:^9}".format(actions[policy[state].argmax()]), end=' ')
        print("")
    print("")

def disp_env_model(env):
    for state in range(env.nS - 1):
        for action in range(env.nA):
            for proba, next_state, reward, done in env.P[state][action]:
                print("state: {}, action: {}, transition proba: {}, next state: {}, reward: {}, done: {}".format(state, action, proba, next_state, reward, done))

def evaluate_policy(env, policy, gamma=1.0):
    a, b = np.eye(env.nS), np.zeros((env.nS))
    for state in range(env.nS - 1):
        for action in range(env.nA):
            pi = policy[state][action]
            for proba, next_state, reward, done in env.P[state][action]:
                a[state, next_state] -= pi * gamma * proba
                b[state] += pi * reward * proba
    v = np.linalg.solve(a, b)
    q = np.zeros((env.nS, env.nA))
    for state in range(env.nS - 1):
        for action in range(env.nA):
            for proba, next_state, reward, done in env.P[state][action]:
                q[state][action] += (proba * (reward + gamma * v[next_state]))
    return v, q

def disp_v_as_map(env, v):
    map_row, map_column = env.shape
    for row in range(map_row):
        for column in range(map_column):
            state = row * map_column + column
            print("{:>10.5f}".format(v[state]), end=' ')
        print("")
    print("")

def test_evaluate_policy(env):
    random_policy = np.random.uniform(size=(env.nS, env.nA))
    random_policy = random_policy / np.sum(random_policy, axis=1)[:, np.newaxis]
    v, q = evaluate_policy(env, random_policy)
    print("random policy state value function = ")
    disp_v_as_map(env, v)
    print("random policy action value function = {}\n".format(q))
    optimal_policy = get_artificial_optimal_policy(env)
    v, q = evaluate_policy(env, optimal_policy)
    print("optimal policy state value function = ")
    disp_v_as_map(env, v)
    print("optimal policy action value function = {}\n".format(q))

def get_optimal_value_func(env, gamma=1.0):
    p = np.zeros((env.nS, env.nA, env.nS))
    r = np.zeros((env.nS, env.nA))
    for state in range(env.nS - 1):
        for action in range(env.nA):
            for proba, next_state, reward, done, in env.P[state][action]:
                p[state, action, next_state] += proba
                r[state, action] += (reward * proba)
    c = np.ones(env.nS)
    a_ub = gamma * p.reshape(-1, env.nS) - np.repeat(np.eye(env.nS), env.nA, axis=0)
    b_ub = -r.reshape(-1, )
    bounds = [(None, None),] * env.nS
    res = scipy.optimize.linprog(c, a_ub, b_ub, bounds=bounds, method='interior-point')
    optimal_v = res.x
    optimal_q = r + gamma * np.dot(p, optimal_v)
    return optimal_v, optimal_q

if __name__ == "__main__":
    env = gym.make('CliffWalking-v0')
    np.set_printoptions(precision=3, suppress=True)

    #get_env_info(env)
    #print('total reward = {}'.format(run_episode(env)))
    #print(get_artificial_optimal_policy(env))
    #print('total reward = {}'.format(run_episode(env, get_artificial_optimal_policy(env))))
    #disp_policy_as_map(env, get_artificial_optimal_policy(env))

    #disp_env_model(env)
    #test_evaluate_policy(env)

    #optimal_v, optimal_q = get_optimal_value_func(env)
    #optimal_policy = np.eye(env.nA)[optimal_q.argmax(axis=1)]   # convert to OneHotkey represent probability
    #disp_policy_as_map(env, optimal_policy)
    #print('total reward = {}'.format(run_episode(env, optimal_policy)))
    #disp_v_as_map(env, optimal_v)

    env.close()
