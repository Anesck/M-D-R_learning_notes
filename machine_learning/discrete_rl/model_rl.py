import numpy as np

# record available mode of cumulative reward
valid_mode = ['t_step', 'discount']

def update_policy(env, Q_line, state, policy):
    max_q = np.max(Q_line)
    if max_q != 0.0:
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
    evaluate policy by caculating state value function(AKA V_table)

arguments:
    env: return from package gym.make()
    policy: a policy for env
    mode:   valid_mode[0]: 't_step'
            valid_mode[1]: 'discount'

    t_tesp: default T-step cumulative rewards arguments
    discount: default gamma-discount cumulative rewards arguments
    evalu_threshod: defaul gamma-dicount cumulative rewards' evaluation threshold
    evalu_iteration: default max gamma-discount cumulative rewards' evaluation iteration times

returns:
    V_table: record state value function
'''
def policy_evaluation(env, policy, mode=valid_mode[1], t_step=10, discount=0.9, evalu_iteration=100, evalu_threshold=1e-10):
    if mode not in valid_mode:
        print("mode error")
        return

    V_table = np.zeros(env.observation_space.n)
    if mode == valid_mode[0]:
        for step in range(1, t_step+1):
            V_table_temp = np.copy(V_table)
            for state in range(env.observation_space.n):
                V_table[state] = sum([policy[state][action] * sum([trans_prob * (1/step*reward + (step-1)/step*V_table_temp[next_state])
                    for trans_prob, next_state, reward, _ in env.P[state][action]])
                    for action in range(env.action_space.n)])
    else:
        for iteration in range(evalu_iteration):
            V_table_temp = np.copy(V_table)
            for state in range(env.observation_space.n):
                V_table[state] = sum([policy[state][action] * sum([trans_prob * (reward + discount*V_table_temp[next_state])
                    for trans_prob, next_state, reward, _ in env.P[state][action]])
                    for action in range(env.action_space.n)])
            V_table_change = V_table - V_table_temp
            if V_table_change[np.argmax(V_table_change)] < evalu_threshold:
                break
    return V_table


'''
usage:
    get policy by policy iteration algorithm

arguments
    env: return from package gym.make()
    mode:   valid_mode[0]: 't_step'
            valid_mode[1]: 'discount'
    algo_iteration: default max algorithm iteration times in case algorithm no stop
    policy: initial policy, if it is None, it will be set to follows the uniform distribution

    t_tesp: default T-step cumulative rewards arguments
    discount: default gamma-discount cumulative rewards arguments
    evalu_threshod: defaul gamma-dicount cumulative rewards' evaluation threshold
    evalu_iteration: default max gamma-discount cumulative rewards' evaluation iteration times

returns:
    policy: the best policy
    Q_table: record state-action value function
    iteration: record algorithm iteration times
    V_table: record state value function
'''
def policy_iteration(env, mode=valid_mode[1], algo_iteration=100, policy=None,
        t_step=10, discount=0.9, evalu_iteration=100, evalu_threshold=1e-10):
    if mode not in valid_mode:
        print("mode error")
        return

    # initialize policy, policy's actions follows the uniform distribution
    if policy is None:
        policy = np.ones([env.observation_space.n, env.action_space.n]) / env.action_space.n
    V_table = np.zeros(env.observation_space.n)
    Q_table = np.zeros([env.observation_space.n, env.action_space.n])
    policy_temp = np.copy(policy)

    for iteration in range(algo_iteration):
        # evaluate the policy and get its V_table
        V_table = policy_evaluation(env, policy, mode, evalu_iteration=evalu_iteration)

        # get the new policy
        for state in range(env.observation_space.n):
            # caculate Q_table by V_table
            for action in range(env.action_space.n):
                if mode == valid_mode[0]:
                    Q_table[state][action] = sum([trans_prob * (1/(t_step+1)*reward + t_step/(t_step+1)*V_table[next_state])
                            for trans_prob, next_state, reward, _ in env.P[state][action]])
                else:
                    Q_table[state][action] = sum([trans_prob * (reward + discount*V_table[next_state])
                            for trans_prob, next_state, reward, _ in env.P[state][action]])

            # update the policy by Q_table
            update_policy(env, Q_table[state, :], state, policy_temp)

        # update policy if unequal else break
        if (policy == policy_temp).all():
            break;
        else:
            policy = np.copy(policy_temp) # policy = update_policy is wrong !!!

    return policy, Q_table, iteration, V_table


'''
usage:
    get policy by value iteration algorithm

arguments:
    env: return from package gym.make()
    mode:   valid_mode[0]: 't_step'
            valid_mode[1]: 'discount'
    algo_iteration: default max algorithm iteration times in case algorithm no stop
    algo_threshold: defaul value iteration algorithm threshold for judging V_table changes
    discount: default gamma-discount cumulative rewards arguments
    policy: initial policy, if it is None, it will be set to follows the uniform distribution

returns:
    policy: the best policy
    Q_table: record state-action value function
    iteration: record algorithm iteration times
    V_table: record state value function
'''
def value_iteration(env, mode=valid_mode[1], algo_iteration=100, algo_threshold=1e-3, discount=0.9, policy=None):
    if mode not in valid_mode:
        print("mode error")
        return

    # initialize policy, policy's actions follows the uniform distribution
    if policy is None:
        policy = np.ones([env.observation_space.n, env.action_space.n]) / env.action_space.n
    V_table = np.zeros(env.observation_space.n)
    Q_table = np.zeros([env.observation_space.n, env.action_space.n])

    # state value function iteration until it's convergent
    for step in range(1, algo_iteration+1):
        V_table_temp = np.copy(V_table)
        if mode == valid_mode[0]:
            for state in range(env.observation_space.n):
                V_table[state] = max([sum([trans_prob * (1/step*reward + (step-1)/step*V_table_temp[next_state])
                    for trans_prob, next_state, reward, _ in env.P[state][action]])
                    for action in range(env.action_space.n)])
        else:
            for state in range(env.observation_space.n):
                V_table[state] = max([sum([trans_prob * (reward + discount*V_table_temp[next_state])
                    for trans_prob, next_state, reward, _ in env.P[state][action]])
                    for action in range(env.action_space.n)])
        V_table_change = V_table - V_table_temp
        if V_table_change[np.argmax(V_table_change)] < algo_threshold:
            break

    iteration = step - 1
    # caculate the policy by V_table
    for state in range(env.observation_space.n):
        for action in range(env.action_space.n):
            if mode == valid_mode[0]:
                Q_table[state][action] = sum([trans_prob * (1/step*reward + (step-1)/step*V_table[next_state])
                        for trans_prob, next_state, reward, _ in env.P[state][action]])
            else:
                Q_table[state][action] = sum([trans_prob * (reward + discount*V_table[next_state])
                        for trans_prob, next_state, reward, _ in env.P[state][action]])
            update_policy(env, Q_table[state, :], state, policy)

    return policy, Q_table, iteration, V_table
