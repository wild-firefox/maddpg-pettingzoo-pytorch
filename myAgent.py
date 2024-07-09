from copy import deepcopy
from typing import List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.optim import Adam


class Agent:
    """Agent that can interact with environment from pettingzoo"""

    def __init__(self, obs_dim, act_dim, global_obs_dim, actor_lr, critic_lr):
        self.actor = PolicyNet(obs_dim, act_dim) ##

        # critic input all the observations and actions #zh-cn:评论家输入所有的观察和行动
        # if there are 3 agents for example, the input for critic is (obs1, obs2, obs3, act1, act2, act3) #zh-cn:例如有3个代理，评论家的输入是（obs1，obs2，obs3，act1，act2，act3）
        self.critic = MLPNetworkWithAttention(global_obs_dim, 1)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)
        self.target_actor = deepcopy(self.actor)
        self.target_critic = deepcopy(self.critic)
        #self.sigma = 0.01 
        self.action_dim = act_dim

    @staticmethod
    def gumbel_softmax(logits, tau=1.0, eps=1e-20): #
        # NOTE that there is a function like this implemented in PyTorch(torch.nn.functional.gumbel_softmax), #zh-cn:请注意，PyTorch中实现了这样一个函数（torch.nn.functional.gumbel_softmax）
        # but as mention in the doc, it may be removed in the future, so i implement it myself #zh-cn:但是如文档中所述，它可能会在将来被删除，所以我自己实现了它
        epsilon = torch.rand_like(logits)
        logits += -torch.log(-torch.log(epsilon + eps) + eps)
        return F.softmax(logits / tau, dim=-1)

    # def action(self, obs, model_out=False): #
    #     # this method is called in the following two cases:
    #     # a) interact with the environment #zh-cn:与环境交互
    #     # b) calculate action when update actor, where input(obs) is sampled from replay buffer with size: #zh-cn:在更新actor时计算动作，其中输入（obs）是从大小为的重放缓冲区中采样的
    #     # torch.Size([batch_size, state_dim]) 

    #     logits = self.actor(obs)  # torch.Size([batch_size, action_size])
    #     # action = self.gumbel_softmax(logits)
    #     action = F.gumbel_softmax(logits, hard=True) #hard=True表示返回one-hot编码
    #     if model_out:
    #         return action, logits
    #     return action
    
    # 这个是探索的动作  #噪音加在main里
    def action(self, obs): # 连续动作域没有logits
        action = self.actor(obs)
        #action = action + self.sigma * torch.randn(self.action_dim) 
        return action

    # def target_action(self, obs):
    #     # when calculate target critic value in MADDPG, #zh-cn:在MADDPG中计算目标评论家价值时，
    #     # we use target actor to get next action given next states, #zh-cn:我们使用目标演员在给定下一个状态时获得下一个动作，
    #     # which is sampled from replay buffer with size torch.Size([batch_size, state_dim]) #zh-cn：它是从大小为的重放缓冲区中采样的

    #     logits = self.target_actor(obs)  # torch.Size([batch_size, action_size])
    #     # action = self.gumbel_softmax(logits)
    #     action = F.gumbel_softmax(logits, hard=True)
    #     return action.squeeze(0).detach()
    # 这个是不探索的动作
    def target_action(self, obs):
        action = self.actor(obs)
        return action.squeeze(0).detach()  ##？？

    def critic_value(self, state_list: List[Tensor], act_list: List[Tensor]):
        x = torch.cat(state_list + act_list, 1)
        return self.critic(x).squeeze(1)  # tensor with a given length
    # 函数一样 给定的参数不一样
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
    def __init__(self, in_dim, out_dim, hidden_dim_1=256, hidden_dim_2=128,non_linear=nn.ReLU()):
        super(MLPNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim_1),
            non_linear,
            nn.Linear(hidden_dim_1, hidden_dim_2),
            non_linear,
            nn.Linear(hidden_dim_2, out_dim),
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
# 和上面一样 但不同实现
class MLPNetwork1(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim_1=256, hidden_dim_2=128,non_linear=nn.ReLU()):
        super(MLPNetwork1, self).__init__()
        self.fc1 = torch.nn.Linear(in_dim, hidden_dim_1)
        self.fc2 = torch.nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc3 = torch.nn.Linear(hidden_dim_2, out_dim)
        
        #根据计算增益
        gain1 = nn.init.calculate_gain('relu')
        #Xavier均匀分布初始化
        torch.nn.init.xavier_uniform_(self.fc1.weight, gain=gain1)
        torch.nn.init.xavier_uniform_(self.fc2.weight, gain=gain1)
        torch.nn.init.xavier_uniform_(self.fc3.weight, gain=gain1)
        #初始化参数
        self.fc1.bias.data.fill_(0.01)
        self.fc2.bias.data.fill_(0.01)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)



class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim_1=256,hidden_dim_2 =128):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim_1)
        self.fc2 = torch.nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc3 = torch.nn.Linear(hidden_dim_2, action_dim)
        '''
        #根据计算增益
        gain1 = nn.init.calculate_gain('relu')
        gain2 = nn.init.calculate_gain('tanh')
        #Xavier均匀分布初始化
        torch.nn.init.xavier_uniform_(self.fc1.weight, gain=gain1)
        torch.nn.init.xavier_uniform_(self.fc2.weight, gain=gain2)
        #初始化参数
        #self.fc1.weight.data.normal_(0, 0.1)
        self.fc1.bias.data.fill_(0.01)
        #self.fc2.weight.data.normal_(0, 0.1)
        self.fc2.bias.data.fill_(0.01)

        #self.action_bound = action_bound  # action_bound是环境可以接受的动作最大值
        '''
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x)) # tanh输出范围是(-1,1)，符合动作的范围 还需要后续clip
    
## 加上注意力机制
class Attention(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(Attention, self).__init__()
        self.query = nn.Linear(in_dim, hidden_dim) #查询
        self.key = nn.Linear(in_dim, hidden_dim)
        self.value = nn.Linear(in_dim, hidden_dim)
    
    def forward(self, x):
        '''
        公式为：Attention(Q, K, V) = softmax(Q*K^T/sqrt(d_k)) * V
        Q: 查询
        K: 键
        V: 值
        d_k: 键的维度 这里用hidden_dim表示 即:K.size(-1)
        对张量 K 进行转置（transpose）。具体来说，这个操作会将张量 K 的第0维和第1维进行交换
        '''
        Q = self.query(x) 
        K = self.key(x)
        V = self.value(x)
        attn_scores = torch.matmul(Q, K.transpose(0, 1)) / torch.sqrt(torch.tensor(K.size(-1), dtype=torch.float32))
        attn_probs = F.softmax(attn_scores, dim=-1)
        attended_values = torch.matmul(attn_probs, V)
        return attended_values

class MLPNetworkWithAttention(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim_1=256, hidden_dim_2=128, attention_dim=32, non_linear=nn.ReLU()):
        super(MLPNetworkWithAttention, self).__init__()
        self.attention = Attention(in_dim, attention_dim)
        self.fc1 = torch.nn.Linear(attention_dim, hidden_dim_1)
        self.fc2 = torch.nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc3 = torch.nn.Linear(hidden_dim_2, out_dim)
        
        # 根据计算增益
        gain1 = nn.init.calculate_gain('relu')
        # Xavier均匀分布初始化
        torch.nn.init.xavier_uniform_(self.fc1.weight, gain=gain1)
        torch.nn.init.xavier_uniform_(self.fc2.weight, gain=gain1)
        torch.nn.init.xavier_uniform_(self.fc3.weight, gain=gain1)
        # 初始化参数
        self.fc1.bias.data.fill_(0.01)
        self.fc2.bias.data.fill_(0.01)
    
    def forward(self, x):
        x = self.attention(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
## 注意力机制改
class MLPNetworkWithAttention1(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim_1=256, hidden_dim_2=128, attention_dim=32, non_linear=nn.ReLU()):
        super(MLPNetworkWithAttention1, self).__init__()
        self.attention = Attention(in_dim, attention_dim)
        self.fc1 = torch.nn.Linear(attention_dim, hidden_dim_1)
        self.fc2 = torch.nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc3 = torch.nn.Linear(hidden_dim_2, out_dim)
        
        # 根据计算增益
        gain1 = nn.init.calculate_gain('relu')
        # Xavier均匀分布初始化
        torch.nn.init.xavier_uniform_(self.fc1.weight, gain=gain1)
        torch.nn.init.xavier_uniform_(self.fc2.weight, gain=gain1)
        torch.nn.init.xavier_uniform_(self.fc3.weight, gain=gain1)
        # 初始化参数
        self.fc1.bias.data.fill_(0.01)
        self.fc2.bias.data.fill_(0.01)
    
    def forward(self, x,agent_id,agents): #x本来为cat后的张量,增加 x,agent_i,agents #agents=Agent() 
        agent_id_list = list(agents.keys())
        agent_id_index = agent_id_list.index(agent_id) #获取agent_id在agents中的索引 按照顺序排
        #agent_n = len(agent_id_list) #智能体数量 #12为state_dim #3*12=36
        temp1=torch.unsqueeze(x[:,12 * agent_id_index:12 * agent_id_index + 12],0).permute(1, 0, 2) #.permute(1, 0, 2)将第0维和第1维进行交换
        temp2=torch.unsqueeze(torch.unsqueeze(x[:, 36 + agent_id_index], 0),0).permute(2, 1, 0) #.permute(2, 1, 0
        e_q = self.embedding(torch.cat((temp1,temp2),2))
        e_k = []
        for j in range(3): # 3个智能体，除了自己
            if j!=agent_id_index:
                temp3=torch.unsqueeze(x[:,5 * j:5 * j + 5],0).permute(1, 0, 2)
                temp4 = torch.unsqueeze(torch.unsqueeze(X[:, 20 + j], 0), 0).permute(2, 1, 0)
                e_k.append(agents[j].critic.embedding(torch.cat((temp3,temp4),2)))



        x = self.attention(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)