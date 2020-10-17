import numpy as np

# record available mode of monte carlo method
valid_mode = ['on_policy', 'off_policy']

def update_policy(env, Q_line, state, policy):
    max_q = np.max(Q_line)
    max_q_indexes = np.where(Q_line == max_q)[0]
    action_prob = 1.0 / len(max_q_indexes)
    for action in range(env.action_space.n):
        if action in max_q_indexes:
            policy[state][action] = action_prob
        else:
            policy[state][action] = 0
    return policy

'''
usage:
    get policy by monte carlo method

arguments:
    env: return from package gym.make()
    mode:   valid_mode[0]: 'on_policy'
            valid_mode[1]: 'off_policy'
    track_branches: number of track samples
    epsilon: epsilon-greed method argument
    policy_steps: the max step of agent will to try
    policy: initial policy, if it is None, it will be set to follows the uniform distribution

returns:
    policy: policy of env besed on track samples
    Q_table: state-action cumulative reward besed on track samples
    track_branches: number of track samples
'''
def monte_carlo(env, mode=valid_mode[0], track_branches=1000, policy_steps=20, epsilon=0.2, policy=None):
    if mode not in valid_mode:
        print("mode error")
        return

    if policy is None:
       policy = np.ones([env.observation_space.n, env.action_space.n]) / env.action_space.n
    Q_table = np.zeros([env.observation_space.n, env.action_space.n])
    # record each state-action's number of reward
    Q_count = np.zeros([env.observation_space.n, env.action_space.n])

    if mode == valid_mode[0]:
        # trackc[i, j, :] record action, reward, next state
        tracks = np.zeros([track_branches, policy_steps, 3])
    else:
        # trackc[i, j, :] record action, reward, next state, probability of equal action of two policy
        tracks = np.zeros([track_branches, policy_steps, 4])

    for i in range(track_branches):
        # set start state
        state = start_state = env.reset()

        # begin to explore a track
        for step in range(policy_steps):
            action = int(np.where(np.array([np.sum(policy[state][:j+1]) for j in range(env.action_space.n)]) >= np.random.random())[0][0])
            if mode == valid_mode[0]:
                # generate epsilon-greed policy's action
                if np.random.random() < epsilon:
                    action = env.action_space.sample()

                # execute the action
                next_state, reward, done, _ = env.step(action)

                # record this step's infomation
                tracks[i, step, :] = action, reward, next_state

                #### the freozen lake's rule
                if done:
                    tracks[i, step+1:, :] = 0, 0, next_state
                    break
            else:
                # the probability of epsilon-greed' action == original action
                prob_action = 1 - epsilon + epsilon / env.action_space.n
                # generate epsilon-greed policy's action
                if np.random.random() < epsilon:
                    action = env.action_space.sample()
                    # the probability of epsilon-greed' action != original action
                    prob_aciton = epsilon / env.action_space.n

                # execute the action generate by epsilon-greed policy
                next_state, reward, done, _ = env.step(action)

                # record this step's infomation
                tracks[i, step, :] = action, reward, next_state, prob_action

                #### the freozen lake's rule
                if done:
                    tracks[i, step+1:, :] = 0, 0, next_state, 1
                    break
            state = next_state

        # use the track to update Q_table
        state = start_state
        for step in range(policy_steps):
            # caculate cumulative reward of each step's state-action in the track
            if mode == valid_mode[0]:
                Reward = 1/(policy_steps-step) * sum([tracks[i, t, 1] for t in range(step, policy_steps)])
            else:
                Reward = 1/(policy_steps-step) * sum([tracks[i, t, 1] for t in range(step, policy_steps)]) * np.prod([1/tracks[i, t, 3] for t in range(step, policy_steps)])

            # get action and update Q_table
            action = int(tracks[i, step, 0])
            Q_table[state][action] = (Q_table[state][action] * Q_count[state][action] + Reward) / (Q_count[state][action] + 1)
            Q_count[state][action] += 1

            # get next state in this track
            state = int(tracks[i, step, 2])

        # according to Q_table, update policy
        for state in range(env.observation_space.n):
            policy = update_policy(env, Q_table[state, :], state, policy)

    return policy, Q_table, track_branches

'''
usage:
    get policy by sarsa method

arguments:
    env: return from package gym.make()
    track_branches: number of track samples
    epsilon: epsilon-greed method argument
    update_step: Temporal Difference(TD) update step width argument
    discount: gamma-discount cumulative reward argument
    policy: initial policy, if it is None, it will be set to follows the uniform distribution

returns:
    policy: policy of env besed on track samples
    Q_table: state-action cumulative reward besed on track samples
    track_branches: number of track samples
'''
def sarsa(env, track_branches=1000, epsilon=0.2, update_step=0.9, discount=0.9, policy=None):
    if policy is None:
       policy = np.ones([env.observation_space.n, env.action_space.n]) / env.action_space.n
    Q_table = np.zeros([env.observation_space.n, env.action_space.n])

    for track in range(track_branches):
        # first state and action, action generated by policy
        state = env.reset()
        action = int(np.where(np.array([np.sum(policy[state][:j+1]) for j in range(env.action_space.n)]) >= np.random.random())[0][0])

        # begin to explore a track
        while True:
            # proceed epsilon-greed policy's action
            next_state, reward, done, _ = env.step(action)

            # next action generated by epsilon-greed policy
            if np.random.random() < epsilon:
                next_action = env.action_space.sample()
            else:
                next_action = int(np.where(np.array([np.sum(policy[next_state][:j+1]) for j in range(env.action_space.n)]) >= np.random.random())[0][0])

            # update Q_table
            Q_table[state][action] += update_step * (reward + discount*Q_table[next_state][next_action] - Q_table[state][action])

            # update action of this state
            policy = update_policy(env, Q_table[state, :], state, policy)

            if done:
                break
            # update state and action
            state = next_state
            action = next_action

    return policy, Q_table, track_branches

'''
usage:
    get policy by Q_learning method

arguments:
    env: return from package gym.make()
    track_branches: number of track samples
    epsilon: epsilon-greed method argument
    update_step: Temporal Difference(TD) update step width argument
    discount: gamma-discount cumulative reward argument
    policy: initial policy, if it is None, it will be set to follows the uniform distribution

returns:
    policy: policy of env besed on track samples
    Q_table: state-action cumulative reward besed on track samples
    track_branches: number of track samples
'''
def Q_learning(env, track_branches=1000, epsilon=0.2, update_step=0.9, discount=0.9, policy=None):
    if policy is None:
       policy = np.ones([env.observation_space.n, env.action_space.n]) / env.action_space.n
    Q_table = np.zeros([env.observation_space.n, env.action_space.n])

    for track in range(track_branches):
        # first state
        state = env.reset()

        # begin to explore a track
        while True:
            # epsilon-greed policy
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.where(np.array([np.sum(policy[state][:j+1]) for j in range(env.action_space.n)]) >= np.random.random())[0][0])

            # proceed epsilon-greed policy's action
            next_state, reward, done, _ = env.step(action)

            # next action generated by original policy
            next_action = int(np.where(np.array([np.sum(policy[next_state][:j+1]) for j in range(env.action_space.n)]) >= np.random.random())[0][0])

            # update Q_table
            Q_table[state][action] += update_step * (reward + discount*Q_table[next_state][next_action] - Q_table[state][action])

            # update action of this state
            policy = update_policy(env, Q_table[state, :], state, policy)

            if done:
                break
            # update state and action
            state = next_state

    return policy, Q_table, track_branches
