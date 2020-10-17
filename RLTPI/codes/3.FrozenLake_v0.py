import gym
import numpy as np

MAPS = {

    "4x4": [

        "SFFH",

        "FFHF",

        "HFFF",

        "FFHG"

    ],
    }

def map_analysis(env):
    env.reset()
    img = env.render("ansi")
    img = img.split('\n')
    del img[0], img[-1]
    img[0] = 'S' + img[0][len(img[0])-len(img[1])+1 : len(img[0])]
    return img

def disp_policy_as_map(env, policy, proba_flag=False):
    actions = ["left/", "down/", "right/", "up/"]
    for i in range(env.observation_space.n):
        action = ""
        if i%4 == 0:
            print("")
        if i == env.observation_space.n-1:
            action += "YES"
        elif map_analysis(env)[i//4][i%4] == 'H':
            action += "X"
        else:
            for j in range(env.action_space.n):
                if policy[i][j] != 0.0:
                    if proba_flag:
                        action += str("{:.2f}/".format(policy[i][j]))
                    else:
                        action += actions[j]
        print("[ {:>2}: {:^22}]".format(i, action), end='')
    print("\n")

def run_episode(env, policy, render=False):
    episode_reward = 0
    state = env.reset()
    while True:
        if render:
            env.render()
        action = np.random.choice(env.action_space.n, p=policy[state])
        next_state, reward, done, _ = env.step(action)
        state = next_state

        episode_reward += reward
        if done:
            break
    return episode_reward

class PolicyIteration():
    def __init__(self, env, gamma=0.99, tolerant=1e-6):
        self.env = env
        self.gamma = gamma
        self.tolerant = tolerant
        self.policy = np.ones((env.observation_space.n, env.action_space.n)) / env.action_space.n
        self.v_table = np.zeros(env.observation_space.n)
        self.q_table = np.zeros((env.observation_space.n, env.action_space.n))
        self.policy_is_optimal = False

    def v2q(self, single_state=None):
        if single_state is not None:
            q_line = np.zeros(self.env.action_space.n)
            for action in range(self.env.action_space.n):
                for proba, next_state, reward, done in self.env.P[single_state][action]:
                    q_line[action] += proba * (reward + self.gamma * self.v_table[next_state] * (1.0 - done))
            return q_line
        else:
            for state in range(self.env.observation_space.n):
                self.q_table[state] = self.v2q(state)

    def evaluate_policy(self):
        self.v_table[:] = 0
        while True:
            delta = 0
            for state in range(self.env.observation_space.n):
                v_new = sum(self.policy[state] * self.v2q(state))
                delta = max(delta, abs(self.v_table[state]-v_new))
                self.v_table[state] = v_new
            if delta < self.tolerant:
                break

    def improve_policy(self):
        self.v2q()
        actions = np.argmax(self.q_table, axis=1)
        policy = np.eye(self.env.observation_space.n, \
                self.env.action_space.n)[actions]
        if (self.policy == policy).all():
            self.policy_is_optimal = True
        else:
            self.policy = policy

    def iterate_policy(self):
        while True:
            self.evaluate_policy()
            self.improve_policy()
            if self.policy_is_optimal:
                break

class ValueIteration():
    def __init__(self, env, gamma=0.99, tolerant=1e-6):
        self.env = env
        self.gamma = gamma
        self.tolerant = tolerant
        self.v_table = np.zeros(env.observation_space.n)
        self.q_table = np.zeros((env.observation_space.n, env.action_space.n))

    def v2q(self, single_state=None):
        if single_state is not None:
            q_line = np.zeros(self.env.action_space.n)
            for action in range(self.env.action_space.n):
                for proba, next_state, reward, done in self.env.P[single_state][action]:
                    q_line[action] += proba * (reward + self.gamma * self.v_table[next_state] * (1.0 - done))
            return q_line
        else:
            for state in range(self.env.observation_space.n):
                self.q_table[state] = self.v2q(state)

    def iterate_value(self):
        while True:
            delta = 0
            for state in range(self.env.observation_space.n):
                v_max = max(self.v2q(state))
                delta = max(delta, abs(self.v_table[state]-v_max))
                self.v_table[state] = v_max
            if delta < self.tolerant:
                break
        self.v2q()
        actions = np.argmax(self.q_table, axis=1)
        self.policy = np.eye(self.env.observation_space.n, \
                self.env.action_space.n)[actions]

if __name__ == "__main__":
    env = gym.make('FrozenLake-v0', desc=None, is_slippery=False)
    #env = gym.make('FrozenLake-v0', desc=MAPS["4x4"], is_slippery=False)
    np.set_printoptions(precision=3, suppress=True)

    #random_policy = np.random.random((env.observation_space.n, env.action_space.n))
    #random_policy = random_policy / random_policy.sum(axis=1)[:, np.newaxis] 
    #run_episode(env, random_policy, render=True)
    #disp_policy_as_map(env, random_policy)
    #disp_policy_as_map(env, random_policy, proba_flag=True)

    '''
    PI = PolicyIteration(env)
    PI.iterate_policy()
    disp_policy_as_map(env, PI.policy)
    print("v_table\n", PI.v_table.reshape(4, 4)[:,:], "\n\nq_table:\n", PI.q_table)
    '''

    '''
    VI = ValueIteration(env)
    VI.iterate_value()
    disp_policy_as_map(env, VI.policy)
    print("v_table\n", VI.v_table.reshape(4, 4)[:,:], "\n\nq_table:\n", VI.q_table)
    '''

    env.close()
