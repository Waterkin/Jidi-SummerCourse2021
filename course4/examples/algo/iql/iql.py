import random
import torch
import torch.nn as nn
import torch.optim as optimizer
import numpy as np

from networks.critic import Critic

import os
from pathlib import Path
import sys

base_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(base_dir))
from common.buffer import Replay_buffer as buffer


def get_trajectory_property():
    return ["action"]


class IQL(object):
    def __init__(self, args, network):

        self.state_dim = args.obs_space
        self.action_dim = args.action_space

        self.hidden_size = args.hidden_size
        self.lr = args.c_lr
        self.buffer_size = args.buffer_capacity
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        # 折扣因子gamma 我想让第t步的reward占现在这一步Q值的0.1 即0.1=gamma^t
        # so, gamma = 0.1 ** (1/t) = 0.8(t=10)
        # self.t=10
        # self.gamma=0.1**(1/self.t)

        if args.given_net:
            self.critic_eval = network  # 方案1：参数共享
        else:
            self.critic_eval = Critic(self.state_dim, self.action_dim, self.hidden_size)  # 方案2 独立训练
        self.critic_target = Critic(self.state_dim, self.action_dim, self.hidden_size)
        self.optimizer = optimizer.Adam(self.critic_eval.parameters(), lr=self.lr)

        # exploration
        self.eps = args.epsilon
        self.eps_end = args.epsilon_end
        # self.eps_delay = 1 / (args.max_episodes * 100)
        # self.eps_delay = 10 / args.max_episodes
        ''''''
        self.eps_delay = 1e-4

        # 更新target网
        self.learn_step_counter = 0
        self.target_replace_iter = args.target_replace

        trajectory_property = get_trajectory_property()
        self.memory = buffer(self.buffer_size, trajectory_property)
        self.memory.init_item_buffers()

        self.update_batch_size = self.batch_size
        self.update_target_replace_iter = self.target_replace_iter
        ''''''
        # 学习率衰减
        # lam = lambda f: 1 - f / (self.learn_step_counter + 1)
        # self.optimizer = optimizer.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lam)
        self.last_q = torch.zeros(self.update_batch_size, 1)  # 为了clip Reward
        self.epsilon = 5  # 把每次Reward的跳变限制在[-5,5]

    def choose_action(self, observation, train=True):
        inference_output = self.inference(observation, train)
        if train:
            self.add_experience(inference_output)
        return inference_output

    def inference(self, observation, train):
        if train:
            ''''''
            # self.eps = max(self.eps_end, self.eps - self.eps_delay) 减法衰减太慢了
            data_length = len(self.memory.item_buffers["action"].data)
            # if data_length % 30 == 0:
            # self.eps = max(self.eps_end, self.eps * (1 - self.eps_delay))  # 指数衰减
            if data_length % 50000 == 0:
                # print('fuck')
                self.eps = max(self.eps_end, self.eps - 0.1)  # 固定步长衰减

            if random.random() < self.eps:
                action = random.randrange(self.action_dim)

            else:
                observation = torch.tensor(observation, dtype=torch.float).view(1, -1)
                action = torch.argmax(self.critic_eval(observation)).item()
        else:
            observation = torch.tensor(observation, dtype=torch.float).view(1, -1)
            action = torch.argmax(self.critic_eval(observation)).item()

        return {"action": action}

    def add_experience(self, output):
        agent_id = 0
        for k, v in output.items():
            self.memory.insert(k, agent_id, v)

    def learn(self):

        data_length = len(self.memory.item_buffers["action"].data)
        # print(data_length)
        if data_length < self.buffer_size:  # 采样一大半数据就可以开始学了
            return

        # 我根据Replay中数据个数来调整 batch size 和 Target网络的更新次数
        '''k = 1 + data_length / self.buffer_size
        self.update_batch_size = int(k * self.batch_size)
        self.update_target_replace_iter = int(k * self.target_replace_iter)'''
        data = self.memory.sample(self.update_batch_size)

        transitions = {
            "o_0": np.array(data['states']),
            "o_next_0": np.array(data['states_next']),
            "r_0": np.array(data['rewards']).reshape(-1, 1),
            "u_0": np.array(data['action']),
            "d_0": np.array(data['dones']).reshape(-1, 1),
        }

        obs = torch.tensor(transitions["o_0"], dtype=torch.float)
        obs_ = torch.tensor(transitions["o_next_0"], dtype=torch.float)
        action = torch.tensor(transitions["u_0"], dtype=torch.long).view(self.update_batch_size,
                                                                         -1)  # batch里每个sample的action
        reward = torch.tensor(transitions["r_0"], dtype=torch.float).squeeze()
        done = torch.tensor(transitions["d_0"], dtype=torch.float).squeeze()

        q_eval = self.critic_eval(obs).gather(1, action)  # 取出每行 index=action的值组成新Tensor
        q_next = self.critic_target(obs_).detach()
        q_target = (reward + self.gamma * q_next.max(1)[0] * (1 - done)).view(self.update_batch_size, 1)
        # print('q_eval:', q_eval.shape())
        # print('q_target:', q_target.shape())
        q_target_mean = torch.mean(q_target)
        q_target_std = torch.std(q_target)
        q_target = q_target / q_target_std
        q_eval = q_eval / torch.std(q_eval)
        # print('q:',q_eval,q_target)
        ''' 
        # 这里试一下Clip的技巧叭
        clip_q = self.last_q + torch.clamp(q_eval - self.last_q, -self.epsilon, self.epsilon)
        # q_max=torch.max((q_eval-q_target)**2,((clip_q-q_target)**2))
        self.last_q = clip_q
        # Q(s,a):=Q(s,a)+alpha(clip(r+gamma argmax(Q(s',a'))-Q(s,a), -1, 1))
        '''
        loss_fn = nn.MSELoss()
        # loss = loss_fn(clip_q, q_target)
        loss = loss_fn(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.learn_step_counter % self.update_target_replace_iter == 0:
            self.critic_target.load_state_dict(self.critic_eval.state_dict())
        self.learn_step_counter += 1

        return loss

    def save(self, save_path, episode, id):
        base_path = os.path.join(save_path, 'trained_model')
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        model_critic_path = os.path.join(base_path, "critic_" + str(id) + "_" + str(episode) + ".pth")
        torch.save(self.critic_eval.state_dict(), model_critic_path)

    def load(self, file):
        self.critic_eval.load_state_dict(torch.load(file))
