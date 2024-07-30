from copy import deepcopy
from typing import List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.optim import Adam

import numpy as np

class Agent:
    """Agent that can interact with environment from pettingzoo"""

    def __init__(self, obs_dim, act_dim, global_obs_act_dim, actor_lr, critic_lr,dim_info,agent_id):
        self.actor = PolicyNet(obs_dim, act_dim) ##

        # critic input all the observations and actions #zh-cn:评论家输入所有的观察和行动
        # if there are 3 agents for example, the input for critic is (obs1, obs2, obs3, act1, act2, act3) #zh-cn:例如有3个代理，评论家的输入是（obs1，obs2，obs3，act1，act2，act3）
        self.critic = MLPNetworkWithAttention2_(global_obs_act_dim, 1 , dim_info,agent_id) #MLPNetwork MLPNetworkWithAttention2
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
    
    # 这个是不探索的动作 #都得加
    def target_action(self, obs):
        action = self.target_actor(obs)   ############!!!这里是target_actor 原来写错了
        # 还得加一个噪音

        return action.detach()  ##移除第0维 大小为1的维度tensor([-2.0882])->-2.0882 #squeeze(0)实际没用上

    def critic_value(self, state_list: List[Tensor], act_list: List[Tensor]):
        x = torch.cat(state_list + act_list, 1) # batch_size * (state_dim + action_dim)
        return self.critic(x).squeeze(1)  # tensor with a given length 移除第1维 大小为1的维度
    
    # 函数不一样、 用的是target_critic
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
    def __init__(self, in_dim, out_dim, dim_info,hidden_dim_1=256, hidden_dim_2=128,non_linear=nn.ReLU()):
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

    def forward(self, x, agent_id, agents):
        return self.net(x)
    
# 和上面一样 但不同实现
class MLPNetwork1(nn.Module):
    def __init__(self, in_dim, out_dim,dim_info, hidden_dim_1=256, hidden_dim_2=128,non_linear=nn.ReLU()):
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
        self.fc3.bias.data.fill_(0.01)
    def forward(self, x,agent_id,agents): #输入维度：batch_size * in_dim 输出维度：batch_size * out_dim
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
    
## 加上注意力机制 之后改为 自注意力机制 
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
        Q = self.query(x) # Q: batch_size * hidden_dim
        K = self.key(x) # K: batch_size * hidden_dim
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
# 弃用上述

## 注意力机制改1 --老师代码版
class Attention1(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(Attention1, self).__init__()
        self.query = nn.Linear(in_dim, hidden_dim, bias = False) #查询
        self.key = nn.Linear(in_dim, hidden_dim, bias = False) #false 好
        #self.value = nn.Linear(in_dim, hidden_dim)
        self.value = nn.Sequential(nn.Linear(in_dim,hidden_dim),nn.LeakyReLU()) # 输出经过激活函数处理
    
    def forward(self, e_q, e_k):  
        '''
        公式为：Attention(Q, K, V) = softmax(Q*K^T/sqrt(d_k)) * V 输出为当前智能体的注意力值
        Q: 查询
        K: 键
        V: 值
        d_k: 键的维度 这里用hidden_dim表示 即:K[0].shape[1]
        e_q: 为batch_size * 1 * in_dim #e_q 为当前状态编码 或者输入batch_size  * in_dim 也可以 view有调整维度的功能
        e_k: 为batch_size * n * in_dim n为【其余】智能体数量 #e_k为其余智能体状态编码
        本质：在其余智能体中找到与当前智能体最相关的智能体
        '''
        Q = self.query(e_q)  #查询当前智能体价值 Q: batch_size * 1 * hidden_dim
        K = self.key(e_k)  #其余智能体的键 K: batch_size * n * hidden_dim
        V = self.value(e_k) #其余智能体的值 V: batch_size * n * hidden_dim
        d_k = K[0].shape[1] #键的维度
        '''
        Q -> batch_size * 1 * hidden_dim 
        K -> batch_size * hidden_dim * n
        Q*K^T -> batch_size * 1 * n
        '''
        fenzi = torch.matmul(Q.view(Q.shape[0], 1, -1),K.permute(0, 2, 1)) #Q*K^T
        atten_scores = fenzi/np.sqrt(d_k) # 维度为 batch_size * 1 * n
        atten_weights = torch.softmax(atten_scores, dim=2) # 维度为 batch_size * 1 * n
        '''
        V -> batch_size * hidden_dim * n
        atten_weights : batch_size * 1 * n  #其余智能体的权重值
        V.permute(0, 2, 1) * atten_weights) : batch_size * hidden_dim * n
        atten_values : batch_size * hidden_dim  #加权求和表示当前智能体的注意力值
        '''
        atten_values = (V.permute(0, 2, 1) * atten_weights).sum(dim=2) ##广播机制会将 atten_weights 的形状扩展为 (batch_size, hidden_dim, n) 等价于
        # atten_values = torch.matmul(V.permute(0, 2, 1), atten_weights.permute(0, 2, 1)).squeeze(2)

        return atten_values #当前智能体的注意力值

class MLPNetworkWithAttention1(nn.Module):
    def __init__(self, in_dim, out_dim, dim_info,hidden_dim_1=256, hidden_dim_2=128, attention_dim=256, non_linear=nn.ReLU()):
        '''
        # in_dim: 为所有智能体状态和动作维度之和 这里是13*3=39 #这里似乎没用到
        # 输入维度为 batch_size * in_dim 输出维度为 batch_size * out_dim
        注意力机制作用：改善了MADDPG中critic输入随智能体数目增大而指数增加的扩展性问题
        '''
        super(MLPNetworkWithAttention1, self).__init__()
        self.attention = Attention1(attention_dim, attention_dim)
        self.fc1 = torch.nn.Linear(2*attention_dim, hidden_dim_1)
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
        self.fc3.bias.data.fill_(0.01)

        # 注意力相关
        self.embedding = nn.Linear(13, attention_dim) #13 为状态维度12+动作维度1
        self.in_fn = nn.BatchNorm1d(2*attention_dim) # BatchNorm1d 只是对每个样本的特征维度归一化 输出和输入维度一样
        self.in_fn.weight.data.fill_(1) #确保在训练开始时，批归一化层不会对输入数据进行任何不必要的缩放和平移，从而保持输入数据的原始分布。这有助于稳定训练过程。
        self.in_fn.bias.data.fill_(0) 
    
    def forward(self, x ,agent_id,agents): # x本来为cat后的张量,增加 x,agent_i,agents #agents=Agent() 
        agent_id_list = list(agents.keys())
        agent_id_index = agent_id_list.index(agent_id) #获取agent_id在agents中的索引 按照顺序排
        agent_n = len(agent_id_list) #智能体数量 #12为state_dim #3*12=36
        '''
        temp1 : permute前 :   1 * batch_size * 12 permute后 : batch_size * 1 * 12
        temp2 : x[:, 36 + agent_id_index] : batch_size #此时只选择第36列 1维张量
        permute前 :  1 * 1 * batch_size   permute后 : batch_size * 1 * 1
        torch.cat((temp1,temp2),2) : batch_size * 1 * 13
        e_q : batch_size * 1 * attention_dim
        【注】我这里动作为列表不是离散值 但是效果一样
        '''
        #print('x:',x[:,12:24].shape)
        temp1 = torch.unsqueeze(x[:,12 * agent_id_index:12 * agent_id_index + 12],0).permute(1, 0, 2) #.permute(1, 0, 2)将第0维和第1维进行交换
        temp2 = torch.unsqueeze(torch.unsqueeze(x[:, 36 + agent_id_index], 0),0).permute(2, 1, 0)  ######
        e_q = self.embedding(torch.cat((temp1,temp2),2))
        '''
        n :【其余】智能体数量
        torch.cat((temp3,temp4),2) : batch_size * 1 * 13
        embedding(torch.cat((temp3,temp4),2)) : batch_size * 1 * attention_dim
        stack: n * batch_size * 1 * attention_dim  #堆叠 : 将多个张量堆叠在一起
        squeeze: n * batch_size * attention_dim #压缩 : 去掉维度为1的维度      
        e_k : batch_size * n * attention_dim 
        '''
        e_k = []
        for j in range(agent_n): # 其余智能体
            if j!=agent_id_index:
                temp3 = torch.unsqueeze(x[:,12 * j:12 * j + 12],0).permute(1, 0, 2)
                temp4 = torch.unsqueeze(torch.unsqueeze(x[:, 36 + j], 0), 0).permute(2, 1, 0)
                e_k.append(agents[agent_id_list[j]].critic.embedding(torch.cat((temp3,temp4),2)))   #agents[j].critic.embedding 在这里使用集中式训练的critic,所以其实这里embedding是一样的

        e_k_s  = torch.squeeze(torch.stack(e_k))  
        e_k = e_k_s.permute(1,0,2)

        atten_values = self.attention(e_q, e_k) #输出 batch_size * attention_dim
        X_in=torch.cat([torch.squeeze(e_q), atten_values], dim=1) # 输出 batch_size * (attention_dim*2)
        h1 = F.relu(self.fc1(self.in_fn(X_in)))
        h2 = F.relu(self.fc2(h1))      ##
        out = (self.fc3(h2))

        return out #输出 batch_size * out_dim

## 注意力机制改1_ --老师代码版+基于MAAC代码修改 + 改成多头
class Attention1_(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads=1):
        super(Attention1_, self).__init__()

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0, "hidden_dim should be divisible by num_heads"

        self.query = nn.Linear(in_dim, hidden_dim, bias = False) #查询
        self.key = nn.Linear(in_dim, hidden_dim, bias = False) #false 好
        self.value = nn.Sequential(nn.Linear(in_dim,hidden_dim),nn.LeakyReLU()) # 输出经过激活函数处理

        self.fc_out = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, e_q, e_k):  
        '''
        公式为：Attention(Q, K, V) = softmax(Q*K^T/sqrt(d_k)) * V 输出为当前智能体的注意力值
        Q: 查询
        K: 键
        V: 值
        d_k: 键的维度 
        e_q: 为batch_size * 1 * in_dim #e_q 为当前状态编码 或者输入batch_size  * in_dim 也可以 view有调整维度的功能
        e_k: 为batch_size * n * in_dim n为【其余】智能体数量 #e_k为其余智能体状态编码
        本质：在其余智能体中找到与当前智能体最相关的智能体
        '''
        Q = self.query(e_q)  #查询当前智能体价值 Q: batch_size * 1 * hidden_dim
        K = self.key(e_k)  #其余智能体的键 K: batch_size * n * hidden_dim
        V = self.value(e_k) #其余智能体的值 V: batch_size * n * hidden_dim
        #d_k = K[0].shape[1] #键的维度 也就是hidden_dim

        # Split the keys, queries and values in num_heads
        Q = Q.view(Q.shape[0], 1, self.num_heads, self.head_dim) #Q: batch_size * 1 * num_heads * head_dim
        K = K.view(K.shape[0], K.shape[1], self.num_heads, self.head_dim) #K: batch_size * n * num_heads * head_dim
        V = V.view(V.shape[0], V.shape[1], self.num_heads, self.head_dim) #V: batch_size * n * num_heads * head_dim

        Q = Q.permute(2, 0, 1, 3)  #Q: num_heads * batch_size * 1 * head_dim
        K = K.permute(2, 0, 1, 3)  #K: num_heads * batch_size * n * head_dim
        V = V.permute(2, 0, 1, 3)  #V: num_heads * batch_size * n * head_dim
        d_k = self.head_dim

        scores = torch.matmul(Q, K.permute(0, 1, 3, 2)) / np.sqrt(d_k) #Q*K^T/sqrt(d_k)  #scores: num_heads * batch_size * 1 * n
        attn_weights = torch.softmax(scores, dim=-1) #softmax(Q*K^T/sqrt(d_k))  #attn_weights: num_heads * batch_size * 1 * n
        attn_values = torch.matmul(attn_weights, V) #attn_weights * V  #attn_values: num_heads * batch_size * 1 * head_dim
        attn_values =  attn_values.permute(1, 2, 0, 3).contiguous() #batch_size * 1 * num_heads * head_dim
        attn_values = attn_values.view(attn_values.shape[0],  attn_values.shape[1], -1) #batch_size * 1 * hidden_dim

        out = self.fc_out(attn_values.squeeze(1))  # batch_size, hidden_dim

        return out #当前智能体的注意力值

class MLPNetworkWithAttention1_(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim_1=256, hidden_dim_2=128, attention_dim=256, num_heads=8):
        '''
        # in_dim: 为所有智能体状态和动作维度之和 这里是13*3=39 #这里似乎没用到
        # 输入维度为 batch_size * in_dim 输出维度为 batch_size * out_dim
        注意力机制作用：改善了MADDPG中critic输入随智能体数目增大而指数增加的扩展性问题
        '''
        super(MLPNetworkWithAttention1_, self).__init__()
        self.attention = Attention1_(attention_dim, attention_dim, num_heads)
        self.fc1 = torch.nn.Linear(2*attention_dim, hidden_dim_1)
        self.fc2 = torch.nn.Linear(hidden_dim_1, 1)
        
        
        # 根据计算增益
        gain1 = nn.init.calculate_gain('leaky_relu')
        # Xavier均匀分布初始化
        torch.nn.init.xavier_uniform_(self.fc1.weight, gain=gain1)
        torch.nn.init.xavier_uniform_(self.fc2.weight, gain=gain1)
        #torch.nn.init.xavier_uniform_(self.fc3.weight, gain=gain1)
        # 初始化参数
        self.fc1.bias.data.fill_(0.01)
        self.fc2.bias.data.fill_(0.01)
        #self.fc3.bias.data.fill_(0.01)

        # 注意力相关
        self.embedding = nn.Linear(13, attention_dim) #13 为状态维度12+动作维度1
        self.in_fn = nn.BatchNorm1d(2*attention_dim,affine=False) # BatchNorm1d 只是对每个样本的特征维度归一化 输出和输入维度一样 =False 不使用可学习参数 只做归一化
        # self.in_fn.weight.data.fill_(1) #确保在训练开始时，批归一化层不会对输入数据进行任何不必要的缩放和平移，从而保持输入数据的原始分布。这有助于稳定训练过程。
        # self.in_fn.bias.data.fill_(0) 
    
    def forward(self, x ,agent_id,agents): # x本来为cat后的张量,增加 x,agent_i,agents #agents=Agent() 
        agent_id_list = list(agents.keys())
        agent_id_index = agent_id_list.index(agent_id) #获取agent_id在agents中的索引 按照顺序排
        agent_n = len(agent_id_list) #智能体数量 #12为state_dim #3*12=36
        '''
        temp1 : permute前 :   1 * batch_size * 12 permute后 : batch_size * 1 * 12
        temp2 : x[:, 36 + agent_id_index] : batch_size #此时只选择第36列 1维张量
        permute前 :  1 * 1 * batch_size   permute后 : batch_size * 1 * 1
        torch.cat((temp1,temp2),2) : batch_size * 1 * 13
        e_q : batch_size * 1 * attention_dim
        【注】我这里动作为列表不是离散值
        '''
        #print('x:',x[:,12:24].shape)
        temp1 = torch.unsqueeze(x[:,12 * agent_id_index:12 * agent_id_index + 12],0).permute(1, 0, 2) #.permute(1, 0, 2)将第0维和第1维进行交换
        temp2 = torch.unsqueeze(torch.unsqueeze(x[:, 36 + agent_id_index], 0),0).permute(2, 1, 0)  ######
        e_q = self.embedding(torch.cat((temp1,temp2),2)) #这个embedding是当前智能体的embedding
        '''
        n :【其余】智能体数量
        torch.cat((temp3,temp4),2) : batch_size * 1 * 13
        embedding(torch.cat((temp3,temp4),2)) : batch_size * 1 * attention_dim
        stack: n * batch_size * 1 * attention_dim  #堆叠 : 将多个张量堆叠在一起
        squeeze: n * batch_size * attention_dim #压缩 : 去掉维度为1的维度      
        e_k : batch_size * n * attention_dim 
        '''
        e_k = []
        for j in range(agent_n): # 其余智能体
            if j!=agent_id_index:
                temp3 = torch.unsqueeze(x[:,12 * j:12 * j + 12],0).permute(1, 0, 2)
                temp4 = torch.unsqueeze(torch.unsqueeze(x[:, 36 + j], 0), 0).permute(2, 1, 0)
                e_k.append(agents[agent_id_list[j]].critic.embedding(torch.cat((temp3,temp4),2)))   #agents[j].critic.embedding 不一样

        e_k_s  = torch.squeeze(torch.stack(e_k))  
        e_k = e_k_s.permute(1,0,2)

        atten_values = self.attention(e_q, e_k) #输出 batch_size * attention_dim
        X_in=torch.cat([torch.squeeze(e_q), atten_values], dim=1) # 输出 batch_size * (attention_dim*2)
        h1 = F.leaky_relu(self.fc1(self.in_fn(X_in)))
        #h2 = F.relu(self.fc2(h1))      ##
        out = (self.fc2(h1))

        return out #输出 batch_size * out_dim


## 注意力机制改2 --Modelling the Dynamic Joint Policy of Teammates with Attention Multi-agent DDPG 2018 论文版
'''
https://github.com/maohangyu/marl_demo/blob/main/C_models.py#L179
这里的多头仅对encoder进行多头处理,关于论文代码里的critic更新还用到了类似于MAAC(2020)的联合损失技术
注：输出的Q不是MADDPG中的Q = Q_i^u(s,a|a_i=u_i(o_i)) u为actor的策略
而是论文中的单个智能体的Q = Q_i^u_i|u_-i(s,a_i)  =  Σa-i∈A-i [u_-i(a_-i|s)] * Q_i^u_i(s,a_i,a_-i)  u_-i为其余智能体的actor策略
其中u_-i(a_-i|s)近似为atten中的权重
Q_i^u_i(s,a_i,a_-i)近似为atten中的值V (使用多头（多个不同动作）和隐藏向量近似)
'''
class Attention2(nn.Module):
    def __init__(self, encoder_input_dim, decoder_input_dim, hidden_dim, head_count):
        super(Attention2, self).__init__()
        self.fc_encoder_input = nn.Linear(encoder_input_dim, hidden_dim)
        self.fc_encoder_heads = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(head_count)])
        self.fc_decoder_input = nn.Linear(decoder_input_dim, hidden_dim)

    def forward(self, encoder_input, decoder_input):
        ''' encoder_input 由所有智能体的状态和当前智能体动作组成，decoder_input 由其余智能体的动作组成'''
        # encoder_input shape: (batch_size, input_dim)
        encoder_h = F.relu(self.fc_encoder_input(encoder_input))
        # encoder_h shape: (batch_size, hidden_dim)

        encoder_heads = torch.stack([F.relu(head(encoder_h)) for head in self.fc_encoder_heads], dim=0)
        # encoder_heads shape: (head_count, batch_size, hidden_dim)

        # decoder_input shape: (batch_size, input_dim)
        decoder_H = F.relu(self.fc_decoder_input(decoder_input))
        # decoder_H shape: (batch_size, hidden_dim)

        ''' enocde_heads 用作键值对 decoder_H 用作查询 '''
        scores = torch.sum(torch.mul(encoder_heads, decoder_H), dim=2)
        # scores shape: (head_count, batch_size)

        attention_weights = F.softmax(scores.permute(1, 0), dim=1).unsqueeze(2)
        # attention_weights shape: (batch_size, head_count, 1)

        contextual_vector = torch.matmul(encoder_heads.permute(1, 2, 0), attention_weights).squeeze()
        # contextual_vector shape: (batch_size, hidden_dim)

        return contextual_vector

class MLPNetworkWithAttention2(nn.Module):
    def __init__(self, in_dim, out_dim,hidden_dim = 128 ,head_count = 8 ):
        super(MLPNetworkWithAttention2, self).__init__()
        #self.args = args # 3为智能体个数 12为状态维度 1为动作维度 
        self.fc_obs = nn.Linear(12, hidden_dim) 
        self.fc_action = nn.Linear(1, hidden_dim)
        self.attention_modules = Attention2(hidden_dim * (3 + 1), hidden_dim * (3 - 1),hidden_dim, head_count) 
        self.fc_qvalue = nn.Linear(hidden_dim, out_dim) 

    def forward(self, x, agent_id, agents):
        agent_id_list = list(agents.keys())
        agent_id_index = agent_id_list.index(agent_id) #获取agent_id在agents中的索引 按照顺序排
        agent_n = len(agent_id_list) #智能体数量3 #12为state_dim #3*12=36
        
        out_obs_list = [F.relu(self.fc_obs(x[:,:12])) , F.relu(self.fc_obs(x[:,12:24])) , F.relu(self.fc_obs(x[:,24:36]))]               
        # out_obs_list shape: [(batch_size, hidden_dim), ...] #即 batch_size * hidden_dim * agent_count

        out_action_list = [F.relu(self.fc_action(x[:,36:37])) , F.relu(self.fc_action(x[:,37:38])) , F.relu(self.fc_action(x[:,38:39]))]
        # out_action_list shape: [(batch_size, hidden_dim), ...]

        encoder_input = torch.cat(out_obs_list + [out_action_list[agent_id_index]], dim=1)
        # encoder_input shape: (batch_size, hidden_dim * (agent_count + 1))

        decoder_input = torch.cat(out_action_list[:agent_id_index] + out_action_list[agent_id_index+1:], dim=1)
        # decoder_input shape: (batch_size, hidden_dim * (agent_count - 1))

        contextual_vector = self.attention_modules(encoder_input, decoder_input)
        # contextual_vector shape: (batch_size, hidden_dim)

        qvalue = self.fc_qvalue(contextual_vector)
        # qvalue shape: (batch_size, 1)

        return qvalue
    
## 注意力机制改2_ --Modelling the Dynamic Joint Policy of Teammates with Attention Multi-agent DDPG 论文 改版
class Attention2_(nn.Module):
    def __init__(self, encoder_input_dim, decoder_input_dim, hidden_dim, head_count):
        super(Attention2_, self).__init__()
        self.fc_encoder_input = nn.Linear(encoder_input_dim, hidden_dim)
        self.fc_encoder_heads = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(head_count)]) ##
        self.fc_decoder_input = nn.Linear(decoder_input_dim, hidden_dim)

    def forward(self, encoder_input, decoder_input):
        ''' encoder_input 由所有智能体的状态和当前智能体动作组成，decoder_input 由其余智能体的动作组成'''
        # encoder_input shape: (batch_size, input_dim)
        encoder_h = F.relu(self.fc_encoder_input(encoder_input))
        # encoder_h shape: (batch_size, hidden_dim)

        encoder_heads = torch.stack([F.relu(head(encoder_h)) for head in self.fc_encoder_heads], dim=0)
        # encoder_heads shape: (head_count, batch_size, hidden_dim)

        # decoder_input shape: (batch_size, input_dim)
        decoder_H = F.relu(self.fc_decoder_input(decoder_input))
        # decoder_H shape: (batch_size, hidden_dim)
        ''' enocde_heads 用作键值对 decoder_H 用作查询 '''
        scores = torch.sum(torch.mul(encoder_heads, decoder_H), dim=2)
        # scores shape: (head_count, batch_size) <- before sum (head_count, batch_size, hidden_dim) 

        attention_weights = F.softmax(scores.permute(1, 0), dim=1).unsqueeze(2)
        # attention_weights shape: (batch_size, head_count, 1)

        contextual_vector = torch.matmul(encoder_heads.permute(1, 2, 0), attention_weights).squeeze()
        # contextual_vector shape: (batch_size, hidden_dim)

        return contextual_vector
    
'''
如果是为了避免计算代价和参数代价的大幅增长 使用hidden_dim = hidden_dim // head_count 
不过好像使用上述 从时间看差不多 从episode上看差点 
'''
class Attention2_2(nn.Module):
    def __init__(self, encoder_input_dim, decoder_input_dim, hidden_dim, head_count):
        super(Attention2_2, self).__init__()
        self.head_count = head_count
        self.head_dim = hidden_dim // head_count
        assert hidden_dim % head_count == 0, "hidden_dim should be divisible by head_count"
        
        self.fc_encoder_input = nn.Linear(encoder_input_dim, self.head_dim )
        self.fc_encoder_heads = nn.ModuleList([nn.Linear(self.head_dim , self.head_dim ) for _ in range(head_count)]) ##
        self.fc_decoder_input = nn.Linear(decoder_input_dim, self.head_dim ) ## 先hidden_dim 再head_dim  不如直接 head_dim

        #self.fc_d = nn.Linear(hidden_dim, self.head_dim)
        self.fc_out = nn.Linear(self.head_dim, hidden_dim)

    def forward(self, encoder_input, decoder_input):
        ''' encoder_input 由所有智能体的状态和当前智能体动作组成，decoder_input 由其余智能体的动作组成'''
        # encoder_input shape: (batch_size, input_dim)
        encoder_h = F.relu(self.fc_encoder_input(encoder_input))
        # encoder_h shape: (batch_size, head_dim)

        encoder_heads = torch.stack([F.relu(head(encoder_h)) for head in self.fc_encoder_heads], dim=0)
        # encoder_heads shape: (head_count, batch_size, head_dim)

        # decoder_input shape: (batch_size, input_dim)
        decoder_H = F.relu(self.fc_decoder_input(decoder_input)) #(batch_size, head_dim)
        ## decoder_H shape: (batch_size, hidden_dim)
        ## decoder_H = self.fc_d(decoder_H)
        # decoder_H shape: (batch_size, head_dim)   
        ''' enocde_heads 用作键值对 decoder_H 用作查询 '''
        scores = torch.sum(torch.mul(encoder_heads, decoder_H), dim=2)
        # scores shape: (head_count, batch_size)

        attention_weights = F.softmax(scores.permute(1, 0), dim=1).unsqueeze(2)
        # attention_weights shape: (batch_size, head_count, 1)

        contextual_vector = torch.matmul(encoder_heads.permute(1, 2, 0), attention_weights).squeeze()
        # contextual_vector shape: (batch_size, head_dim)

        contextual_vector = self.fc_out(contextual_vector)
        # contextual_vector shape: (batch_size, hidden_dim)

        return contextual_vector

class MLPNetworkWithAttention2_(nn.Module):
    def __init__(self, in_dim, out_dim,dim_info,agent_id,hidden_dim = 128 ,head_count = 8 ):
        '''
        在Attention2中 hidden_dim = 128 ,head_count = 8  效果最好 在3v3的环境中
        测试 256
        '''
        super(MLPNetworkWithAttention2_, self).__init__()
        '''
        #self.args = args # 3为智能体个数 12为状态维度 1为动作维度 
        self.fc_obs = nn.Linear(12, hidden_dim) 
        self.fc_action = nn.Linear(1, hidden_dim)
        '''
        self.attention_modules = Attention2_(hidden_dim , hidden_dim ,hidden_dim, head_count) 
        self.fc_qvalue = nn.Linear(hidden_dim, out_dim) 

        self.dim_info = dim_info #dim_info = {'agent_id':[obs_dim, act_dim],}
        #所有智能体的状态和当前智能体动作 维度
        self.encoder_input_dim = sum([dim_info[agent_id_][0] for agent_id_ in dim_info.keys()]) + dim_info[agent_id][1]
        self.fc1 = torch.nn.Linear(self.encoder_input_dim, hidden_dim)
        #其余智能体的动作 维度
        self.decoder_input_dim = sum([dim_info[agent_id_][1] for agent_id_ in dim_info.keys() if agent_id_ != agent_id])
        self.fc2 = torch.nn.Linear(self.decoder_input_dim, hidden_dim)

        self.fc3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = torch.nn.Linear(hidden_dim, hidden_dim)

        # 所有智能体的状态的维度
        self.all_obs_dim = sum([dim_info[agent_id_][0] for agent_id_ in dim_info.keys()])
        # 所有智能体的状态+动作维度
        self.all_obs_act_dim = in_dim

        # 动作维度的列号
        d = [dim_info[agent_id_][1] for agent_id_ in dim_info.keys()] # [1,2,1] # [1,1,1]
        c_num = 0
        self.d = [0]+[c_num := c_num + i for i in d] # [0,1,3,4] # [0,1,2,3]

        

    def forward(self, x,agent_id,agents):
        agent_id_list = list(agents.keys())
        agent_id_index = agent_id_list.index(agent_id) #获取agent_id在agents中的索引 按照顺序排
        agent_n = len(agent_id_list) #智能体数量 #12为state_dim #3*12=36

        '''改
        out_obs_list = [F.relu(self.fc_obs(x[:,:12])) , F.relu(self.fc_obs(x[:,12:24])) , F.relu(self.fc_obs(x[:,24:36]))]               
        # out_obs_list shape: [(batch_size, hidden_dim), ...] #即 batch_size * hidden_dim * agent_count
        out_action_list = [F.relu(self.fc_action(x[:,36:37])) , F.relu(self.fc_action(x[:,37:38])) , F.relu(self.fc_action(x[:,38:39]))]
        # out_action_list shape: [(batch_size, hidden_dim), ...]
        encoder_input = torch.cat(out_obs_list + [out_action_list[agent_id_index]], dim=1)
        # encoder_input shape: (batch_size, hidden_dim * (agent_count + 1))
        decoder_input = torch.cat(out_action_list[:agent_id_index] + out_action_list[agent_id_index+1:], dim=1)
        # decoder_input shape: (batch_size, hidden_dim * (agent_count - 1))
        '''

        #encoder_input = self.fc1(x[:,:37]) 
        #decoder_input = self.fc2(x[:,37:39]) ##??搞错了 效果还挺好？不如下面好
        
        #action_list = [x[:,36:37],x[:,37:38],x[:,38:39]]
        # 所有智能体的动作对应列
        action_list = x[:,self.all_obs_dim:self.all_obs_act_dim]
        action_list = [action_list[:,self.d[i]:self.d[i+1]] for i in range(len(self.d)-1)]
        encoder_input = self.fc1(torch.cat((x[:,:self.all_obs_dim],action_list[agent_id_index]),1)) #batch_size * 37 -> batch_size * hidden_dim
        decoder_input = self.fc2(torch.cat((action_list[:agent_id_index]+action_list[agent_id_index+1:]),1)) #batch_size * 2 -> batch_size * hidden_dim

        # 要满足 encoder_input shape: (batch_size, hidden_dim) decoder_input shape: (batch_size, hidden_dim) 
        contextual_vector = self.attention_modules(encoder_input, decoder_input)
        # contextual_vector shape: (batch_size, hidden_dim)
        t1 = F.relu(self.fc3(contextual_vector))
        #t = F.relu(self.fc4(t1))

        qvalue = self.fc_qvalue(t1)
        # qvalue shape: (batch_size, 1)

        return qvalue
    
## 注意力机制改3 --MAAC
#https://github.com/shariqiqbal2810/MAAC/blob/master/utils/critics.py#L8
class Attention3(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads):
        super(Attention3, self).__init__()

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0, "hidden_dim should be divisible by num_heads"

        self.query = nn.Linear(in_dim, hidden_dim, bias = False) #查询
        self.key = nn.Linear(in_dim, hidden_dim, bias = False) #false 好
        self.value = nn.Sequential(nn.Linear(in_dim,hidden_dim),nn.LeakyReLU()) # 输出经过激活函数处理

        self.fc_out = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, e_q, e_k):  
        '''
        公式为：Attention(Q, K, V) = softmax(Q*K^T/sqrt(d_k)) * V 输出为当前智能体的注意力值
        Q: 查询
        K: 键
        V: 值
        d_k: 键的维度 
        e_q: 为batch_size * 1 * in_dim #e_q 为当前状态编码 或者输入batch_size  * in_dim 也可以 view有调整维度的功能
        e_k: 为batch_size * n * in_dim n为【其余】智能体数量 #e_k为其余智能体状态编码
        本质：在其余智能体中找到与当前智能体最相关的智能体
        '''
        Q = F.leaky_relu(self.query(e_q))  #查询当前智能体价值 Q: batch_size * 1 * hidden_dim
        K = F.leaky_relu(self.key(e_k))  #其余智能体的键 K: batch_size * n * hidden_dim
        V = F.leaky_relu(self.value(e_k)) #其余智能体的值 V: batch_size * n * hidden_dim
        #d_k = K[0].shape[1] #键的维度 也就是hidden_dim

        # Split the keys, queries and values in num_heads
        Q = Q.view(Q.shape[0], 1, self.num_heads, self.head_dim) #Q: batch_size * 1 * num_heads * head_dim
        K = K.view(K.shape[0], K.shape[1], self.num_heads, self.head_dim) #K: batch_size * n * num_heads * head_dim
        V = V.view(V.shape[0], V.shape[1], self.num_heads, self.head_dim) #V: batch_size * n * num_heads * head_dim

        Q = Q.permute(2, 0, 1, 3)  #Q: num_heads * batch_size * 1 * head_dim
        K = K.permute(2, 0, 1, 3)  #K: num_heads * batch_size * n * head_dim
        V = V.permute(2, 0, 1, 3)  #V: num_heads * batch_size * n * head_dim
        d_k = self.head_dim

        scores = torch.matmul(Q, K.permute(0, 1, 3, 2)) / np.sqrt(d_k) #Q*K^T/sqrt(d_k)  #scores: num_heads * batch_size * 1 * n
        attn_weights = torch.softmax(scores, dim=-1) #softmax(Q*K^T/sqrt(d_k))  #attn_weights: num_heads * batch_size * 1 * n
        attn_values = torch.matmul(attn_weights, V) #attn_weights * V  #attn_values: num_heads * batch_size * 1 * head_dim
        attn_values =  attn_values.permute(1, 2, 0, 3).contiguous() #batch_size * 1 * num_heads * head_dim
        attn_values = attn_values.view(attn_values.shape[0],  attn_values.shape[1], -1) #batch_size * 1 * hidden_dim

        out = self.fc_out(attn_values.squeeze(1))  # batch_size, hidden_dim

        return out #当前智能体的注意力值

class MLPNetworkWithAttention3(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim_1=256, hidden_dim_2=128, attention_dim=256, num_heads=4,norm_in=True):
        '''
        # in_dim: 为所有智能体状态和动作维度之和 这里是13*3=39 #这里似乎没用到
        # 输入维度为 batch_size * in_dim 输出维度为 batch_size * out_dim
        注意力机制作用：改善了MADDPG中critic输入随智能体数目增大而指数增加的扩展性问题
        '''
        super(MLPNetworkWithAttention3, self).__init__()
        self.attention = Attention3(attention_dim, attention_dim, num_heads)
        self.fc1 = torch.nn.Linear(2*attention_dim, hidden_dim_1)
        self.fc2 = torch.nn.Linear(hidden_dim_1, out_dim)
        
        #其他
        self.norm_in = norm_in
        self.batchnorm = nn.BatchNorm1d(13,affine=False) 
        
        ''''
        # 根据计算增益
        gain1 = nn.init.calculate_gain('leaky_relu')
        # Xavier均匀分布初始化
        torch.nn.init.xavier_uniform_(self.fc1.weight, gain=gain1)
        torch.nn.init.xavier_uniform_(self.fc2.weight, gain=gain1)
        # 初始化参数
        self.fc1.bias.data.fill_(0.01)
        self.fc2.bias.data.fill_(0.01)
        '''

        # 注意力相关
        self.embedding = nn.Linear(13, attention_dim) #13 为状态维度12+动作维度1
        #self.in_fn = nn.BatchNorm1d(2*attention_dim,affine=False) # BatchNorm1d 只是对每个样本的特征维度归一化 输出和输入维度一样 =False 不使用可学习参数 只做归一化
        # self.in_fn.weight.data.fill_(1) #确保在训练开始时，批归一化层不会对输入数据进行任何不必要的缩放和平移，从而保持输入数据的原始分布。这有助于稳定训练过程。
        # self.in_fn.bias.data.fill_(0) 
    
    def forward(self, x ,agent_id,agents): # x本来为cat后的张量,增加 x,agent_i,agents #agents=Agent() 
        agent_id_list = list(agents.keys())
        agent_id_index = agent_id_list.index(agent_id) #获取agent_id在agents中的索引 按照顺序排
        agent_n = len(agent_id_list) #智能体数量 #12为state_dim #3*12=36
        '''
        temp1 : permute前 :   1 * batch_size * 12 permute后 : batch_size * 1 * 12
        temp2 : x[:, 36 + agent_id_index] : batch_size #此时只选择第36列 1维张量
        permute前 :  1 * 1 * batch_size   permute后 : batch_size * 1 * 1
        torch.cat((temp1,temp2),2) : batch_size * 1 * 13
        e_q : batch_size * 1 * attention_dim
        【注】我这里动作为列表不是离散值
        '''
        #print('x:',x[:,12:24].shape)
        temp1 = torch.unsqueeze(x[:,12 * agent_id_index:12 * agent_id_index + 12],0).permute(1, 0, 2) #.permute(1, 0, 2)将第0维和第1维进行交换
        temp2 = torch.unsqueeze(torch.unsqueeze(x[:, 36 + agent_id_index], 0),0).permute(2, 1, 0)  ######
        cat_temp = torch.cat((temp1,temp2),2)
        if self.norm_in:
            # batchborm 只接受2维张量
            cat_temp = self.batchnorm(torch.squeeze(cat_temp))
            cat_temp = torch.unsqueeze(cat_temp,1)
        e_q = F.leaky_relu(self.embedding(cat_temp)) #这个embedding是当前智能体的embedding #这里s 和 a已经是归一化过的 所以这里不用加BatchNorm1d 

        '''
        n :【其余】智能体数量
        torch.cat((temp3,temp4),2) : batch_size * 1 * 13
        embedding(torch.cat((temp3,temp4),2)) : batch_size * 1 * attention_dim
        stack: n * batch_size * 1 * attention_dim  #堆叠 : 将多个张量堆叠在一起
        squeeze: n * batch_size * attention_dim #压缩 : 去掉维度为1的维度      
        e_k : batch_size * n * attention_dim 
        '''
        e_k = []
        for j in range(agent_n): # 其余智能体
            if j!=agent_id_index:
                temp3 = torch.unsqueeze(x[:,12 * j:12 * j + 12],0).permute(1, 0, 2)
                temp4 = torch.unsqueeze(torch.unsqueeze(x[:, 36 + j], 0), 0).permute(2, 1, 0)
                cat_temp_ = torch.cat((temp3,temp4),2)
                if self.norm_in:
                    cat_temp_ = self.batchnorm(torch.squeeze(cat_temp_))
                    cat_temp_ = torch.unsqueeze(cat_temp_,1)
                e_k.append(F.leaky_relu(agents[agent_id_list[j]].critic.embedding(cat_temp_)))   #agents[j].critic.embedding 不一样

        e_k_s  = torch.squeeze(torch.stack(e_k))  
        e_k = e_k_s.permute(1,0,2)
        atten_values = self.attention(e_q, e_k) #输出 batch_size * attention_dim
        X_in=torch.cat([torch.squeeze(e_q), atten_values], dim=1) # 输出 batch_size * (attention_dim*2)
        h1 = F.leaky_relu(self.fc1(X_in))
        out = (self.fc2(h1))

        return out #输出 batch_size * out_dim