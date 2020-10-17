import gym
import numpy as np
import model_rl as mrl
import model_free_rl as mfrl
import value_func_approxi as vfa
import MySQLdb as sql

MAPS = {

    "4x4": [

        "SFFH",

        "FFHF",

        "HFFF",

        "FHFG"

    ],
    }

def map_analysis(env):
    env.reset()
    img = env.render("ansi")
    img = img.split('\n')
    del img[0], img[-1]
    img[0] = 'S' + img[0][len(img[0])-len(img[1])+1 : len(img[0])]
    return img

def show_message(env, mode, policy, Q_table, iteration=0, V_table=None, size=4):
    print("*" * 100, "\n mode:\t{}\t\t iteration times:\t{}\n".format(mode, iteration))
    for i in range(env.observation_space.n):
        action = ""
        if i%size == 0:
            print("")
        if i == env.observation_space.n-1:
            action += "YES"
        elif  map_analysis(env)[i//4][i%4] is 'H':
            action += "X"
        else:
            for j in range(env.action_space.n):
                if policy[i][j] != 0.0:
                    if j == 0:
                        action += "left/"
                    if j == 1:
                        action += "down/"
                    if j == 2:
                        action += "right/"
                    if j == 3:
                        action += "up/"
        print("[ {:^3}: {:^20}]".format(i, action), end='')
    if V_table is not None:
        print("\n\n V_table:\n{}\n".format(np.reshape(V_table, (size, size))))
    print("\n policy table:\n{}\n\n Q_table:\n{}\n\n".format(policy, Q_table))

def save_Qtable(Q_table):
    db = sql.connect(user='root', passwd='578449', db='mydb');
    cur = db.cursor();
    cur.execute("drop table Q_table")
    cur.execute("create table Q_table( \
            state_id int not null auto_increment, \
            left_ float, \
            down_ float, \
            right_ float, \
            up_ float, \
            primary key (state_id)) \
            charset=utf8;");
    for i in Q_table:
        cur.execute("insert into Q_table (left_, down_, right_, up_) values ({}, {}, {}, {})".format(*(i)))
    cur.close()
    db.commit()
    db.close()

if __name__ == '__main__':
#    env = gym.make('FrozenLake-v0', desc=None)
    env = gym.make('FrozenLake-v0', desc=None, is_slippery=False)
    env.render()

    # model rl
    rule = 1
    algo_type = 1
    if rule == 1:
        if algo_type == 1:
            policy, Q_table, iteration, V_table = mrl.policy_iteration(env)
        elif algo_type == 2:
            policy, Q_table, iteration, V_table = mrl.value_iteration(env)
        show_message(env, 'discount', policy, Q_table, iteration, V_table)
    elif rule == 2:
        mode = 't_step'
        # change the victory state reward
        for k, v in env.P[env.observation_space.n-1].items():
            temp = list(v[0])
            temp[2] = 1
            env.P[env.observation_space.n-1][k] = [tuple(temp)]
        if algo_type == 1:
            policy, Q_table, iteration, V_table = mrl.policy_iteration(env, mode)
        elif algo_type == 2:
            policy, Q_table, iteration, V_table = mrl.value_iteration(env, mode)
        show_message(env, mode, policy, Q_table, iteration, V_table)


    # model free rl
    algo_type = 0
    if algo_type == 1:
        policy, Q_table, iteration = mfrl.monte_carlo(env)
        V_table = mrl.policy_evaluation(env, policy)
        show_message(env, 'on_policy', policy, Q_table, iteration, V_table)
    elif algo_type == 2:
        policy, Q_table, iteration = mfrl.monte_carlo(env, mode='off_policy')
        V_table = mrl.policy_evaluation(env, policy)
        show_message(env, 'off_policy', policy, Q_table, iteration, V_table)
    elif algo_type == 3:
        policy, Q_table, iteration = mfrl.sarsa(env)
        V_table = mrl.policy_evaluation(env, policy)
        show_message(env, 'sarsa', policy, Q_table, iteration, V_table)
    elif algo_type == 4:
        policy, Q_table, iteration = mfrl.Q_learning(env)
        V_table = mrl.policy_evaluation(env, policy)
        show_message(env, 'Q_learning', policy, Q_table, iteration, V_table)

    # value function approximate
    algo_type = 0
    if algo_type == 1:
        policy, Q_table, iteration, plt = vfa.linear_approxi_sarsa(env)
        V_table = mrl.policy_evaluation(env, policy)
        show_message(env, 'value_func_approxi', policy, Q_table, iteration, V_table)
        plt.show()

    env.close()
