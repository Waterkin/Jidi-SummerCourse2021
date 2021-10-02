# # This is homework.
# # Load your model and submit this to Jidi


import numpy as np
import os

# todo
# Once start to train, u can get saved model. Here we just say it is q_table.pth.
q_table = os.path.dirname(os.path.abspath(__file__)) + '/q_1000.pth'
q_values = np.loadtxt(q_table, delimiter=",")


def action_from_algo_to_env(joint_action):
    joint_action_ = []
    for a in range(1):
        action_a = joint_action
        each = [0] * 4
        each[action_a] = 1
        joint_action_.append(each)
    return joint_action_


# todo
def behaviour_policy(q):
    # pass
    return epsilon_greedy(q)


# todo
def epsilon_greedy(q_values):
    # pass
    return np.argmax(q_values)


# todo
def my_controller(observation, action_space, is_act_continuous=False):
    obs = observation['obs']
    # pass
    action = behaviour_policy(q_values[obs, :])  # 对obs对应的q值使用behavior policy
    action_onehot = action_from_algo_to_env(action)
    return action_onehot
