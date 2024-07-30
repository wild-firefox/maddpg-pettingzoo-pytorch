from copy import deepcopy
from typing import List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.optim import Adam


class Agent:
    """Agent that can interact with environment from pettingzoo"""

    def __init__(self, obs_dim, act_dim, global_obs_dim, actor_lr, critic_lr):
        self.actor = MLPNetwork(obs_dim, act_dim)

        # critic input all the observations and actions #zh-cn:评论家输入所有的观察和行动
        # if there are 3 agents for example, the input for critic is (obs1, obs2, obs3, act1, act2, act3) #zh-cn:例如有3个代理，评论家的输入是（obs1，obs2，obs3，act1，act2，act3）
        self.critic = MLPNetwork(global_obs_dim, 1)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)
        self.target_actor = deepcopy(self.actor)
        self.target_critic = deepcopy(self.critic)

    @staticmethod
    def gumbel_softmax(logits, tau=1.0, eps=1e-20): #
        # NOTE that there is a function like this implemented in PyTorch(torch.nn.functional.gumbel_softmax), #zh-cn:请注意，PyTorch中实现了这样一个函数（torch.nn.functional.gumbel_softmax）
        # but as mention in the doc, it may be removed in the future, so i implement it myself #zh-cn:但是如文档中所述，它可能会在将来被删除，所以我自己实现了它
        epsilon = torch.rand_like(logits) #torch.rand_like(input)返回一个与输入张量input相同大小的张量，其中元素是从区间[0, 1)的均匀分布中抽取的。
        logits += -torch.log(-torch.log(epsilon + eps) + eps)
        return F.softmax(logits / tau, dim=-1)

    def action(self, obs, model_out=False): #
        # this method is called in the following two cases:
        # a) interact with the environment #zh-cn:与环境交互
        # b) calculate action when update actor, where input(obs) is sampled from replay buffer with size: #zh-cn:在更新actor时计算动作，其中输入（obs）是从大小为的重放缓冲区中采样的
        # torch.Size([batch_size, state_dim]) 

        logits = self.actor(obs)  # torch.Size([batch_size, action_size])
        # action = self.gumbel_softmax(logits)
        action = F.gumbel_softmax(logits, hard=True) #hard=True表示返回one-hot编码 
        if model_out:
            return action, logits
        return action 

    def target_action(self, obs):
        # when calculate target critic value in MADDPG, #zh-cn:在MADDPG中计算目标评论家价值时，
        # we use target actor to get next action given next states, #zh-cn:我们使用目标演员在给定下一个状态时获得下一个动作，
        # which is sampled from replay buffer with size torch.Size([batch_size, state_dim]) #zh-cn：它是从大小为的重放缓冲区中采样的

        logits = self.target_actor(obs)  # torch.Size([batch_size, action_size])  ################！！！ 这里是target_actor
        # action = self.gumbel_softmax(logits)
        action = F.gumbel_softmax(logits, hard=True) #hard=True表示返回one-hot编码  #False表示返回概率分布 #batch_size, action_size
        return action.squeeze(0).detach()  #squeeze()函数的作用是对数据的维度进行压缩。a.squeeze(0)表示压缩第0维的维度，a.squeeze(1)表示压缩第1维的维度，以此类推。当然，a.squeeze()表示压缩所有的维度。detach()返回一个新的Variable，从当前计算图中分离下来的，但是仍指向原变量的存放位置，不同之处只是requires_grad为false，得到的这个Variable永远不需要计算其梯度，不具有grad。
                                           # 实际这里没用上squeeze
    def critic_value(self, state_list: List[Tensor], act_list: List[Tensor]):
        x = torch.cat(state_list + act_list, 1)
        return self.critic(x).squeeze(1)  # tensor with a given length

    def target_critic_value(self, state_list: List[Tensor], act_list: List[Tensor]):
        x = torch.cat(state_list + act_list, 1)
        return self.target_critic(x).squeeze(1)  # tensor with a given length

    def update_actor(self, loss):
        self.actor_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()

    def update_critic(self, loss):
        self.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()


class MLPNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=64, non_linear=nn.ReLU()):
        super(MLPNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            non_linear,
            nn.Linear(hidden_dim, hidden_dim),
            non_linear,
            nn.Linear(hidden_dim, out_dim),
        ).apply(self.init) #apply(self.init)是在初始化模块的权重和偏置时调用init方法

    @staticmethod
    def init(m):
        """init parameter of the module"""
        gain = nn.init.calculate_gain('relu') #zh-cn:计算增益
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=gain)#这行代码使用 Xavier 均匀分布初始化方法来初始化模块的权重（m.weight）。Xavier 初始化方法旨在使得网络各层的激活值和梯度的方差在传播过程中保持一致，有助于加速网络的收敛。gain 参数是根据 ReLU 激活函数的特性调整的。
            m.bias.data.fill_(0.01) #zh-cn:这行代码使用常数 0.01 来初始化模块的偏置（m.bias）。

    def forward(self, x):
        return self.net(x)
