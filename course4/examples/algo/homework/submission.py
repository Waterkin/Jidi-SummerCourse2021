# # This is homework.
# # Load your model and submit this to Jidi
import numpy as np
import torch
import os
import torch.optim as optimizer
# load critic
import importlib
from pathlib import Path
import sys

base_dir = Path(__file__).resolve().parent
sys.path.append(str(base_dir))
from critic import Critic

''' Block1 : IQL Build
    ①能初始化一个指定大小的Critic网络
    ②能加载训练过的网络
    ③能用这个网络选择动作
'''


class MultiRLAgents:
    def __init__(self):
        self.agents = list()
        self.n_player = 2

        for i in range(self.n_player):
            agent = IQL()
            self.agents.append(agent)

    def action_from_algo_to_env(self, joint_action):
        joint_action_ = []
        for a in range(1):
            action_a = joint_action
            each = [0] * 4
            each[action_a] = 1
            joint_action_.append(each)
        return joint_action_

    # ======================================= inference =============================================
    def choose_action_to_env(self, observation, id):
        obs_copy = observation.copy()
        action_from_algo = self.agents[id].choose_action(obs_copy)
        action_to_env = self.action_from_algo_to_env(action_from_algo)
        return action_to_env

    def load(self, file_list):
        # 加载训练好的网络
        for index, agent in enumerate(self.agents):
            agent.load(file_list[index])


# TODO
class IQL:
    def __init__(self):
        self.state_dim = 18
        self.action_dim = 4
        self.hidden_size = 64
        self.num_hidden_layer=2
        self.critic_eval = Critic(self.state_dim, self.action_dim, self.hidden_size, self.num_hidden_layer)

    def choose_action(self, observation):
        observation = torch.tensor(observation, dtype=torch.float).view(
            1, -1
        )
        action = torch.argmax(self.critic_eval(observation)).item()

        return action

    def load(self, file):
        # 加载网络参数
        self.critic_eval.load_state_dict(torch.load(file))


''' Block2 : State to Observations '''


def get_surrounding(state, width, height, x, y):
    surrounding = [state[(y - 1) % height][x],  # up
                   state[(y + 1) % height][x],  # down
                   state[y][(x - 1) % width],  # left
                   state[y][(x + 1) % width]]  # right

    return surrounding


def make_grid_map(board_width, board_height, beans_positions: list, snakes_positions: dict):
    snakes_map = [[[0] for _ in range(board_width)] for _ in range(board_height)]
    for index, pos in snakes_positions.items():
        for p in pos:
            snakes_map[p[0]][p[1]][0] = index

    for bean in beans_positions:
        snakes_map[bean[0]][bean[1]][0] = 1

    return snakes_map

# Self position:        0:head_x; 1:head_y
# Head surroundings:    2:head_up; 3:head_down; 4:head_left; 5:head_right
# Beans positions:      (6, 7) (8, 9) (10, 11) (12, 13) (14, 15)
# Other snake positions: (16, 17) (18, 19) (20, 21) (22, 23) (24, 25) -- (other_x - self_x, other_y - self_y)
def get_observations(state, id, obs_dim):
    state_copy = state.copy()
    board_width = state_copy['board_width']
    board_height = state_copy['board_height']
    beans_positions = state_copy[1]
    snakes_positions = {key: state_copy[key] for key in state_copy.keys() & {2, 3, 4, 5, 6}}
    snakes_positions_list = []
    for key, value in snakes_positions.items():
        snakes_positions_list.append(value)
    snake_map = make_grid_map(board_width, board_height, beans_positions, snakes_positions)
    state = np.array(snake_map)
    state = np.squeeze(snake_map, axis=2)

    observations = np.zeros((1, obs_dim))  # todo
    snakes_position = np.array(snakes_positions_list, dtype=object)
    beans_position = np.array(beans_positions, dtype=object).flatten()
    agents_index = [id]
    for i, element in enumerate(agents_index):
        # # self head position
        observations[i][:2] = snakes_positions_list[element][0][:]

        # head surroundings
        head_x = snakes_positions_list[element][0][1]
        head_y = snakes_positions_list[element][0][0]

        head_surrounding = get_surrounding(state, board_width, board_height, head_x, head_y)
        observations[i][2:6] = head_surrounding[:]

        # beans positions
        observations[i][6:16] = beans_position[:]

        # other snake positions # todo: to check
        snake_heads = np.array([snake[0] for snake in snakes_position])
        snake_heads = np.delete(snake_heads, element, 0)
        observations[i][16:] = snake_heads.flatten()[:]
    return observations.squeeze().tolist()


''' Block3 : Load your model 这里每次提交都要改一下！ '''
critic_net_0 = os.path.dirname(os.path.abspath(__file__)) + '/critic_0_28000.pth' #run33-25k
critic_net_1 = os.path.dirname(os.path.abspath(__file__)) + '/critic_1_28000.pth'
# 不共享网络就加载不一样的，共享就加载一样的
critic_list = [critic_net_0, critic_net_1]

agent = MultiRLAgents()
agent.load(critic_list)

n_player = 2


# todo 对一条蛇的观测 输出这条蛇的动作
def my_controller(observation, action_space, is_act_continuous=False):
    # 输入observation 输出action
    obs = observation
    o_index = obs['controlled_snake_index']
    o_index -= 2
    obs = get_observations(observation, o_index, 18)
    action_ = agent.choose_action_to_env(obs, o_index)
    return action_
