import argparse
import os
# 设置OMP_WAIT_POLICY为PASSIVE，让等待的线程不消耗CPU资源 #确保在pytorch前设置
os.environ['OMP_WAIT_POLICY'] = 'PASSIVE' #
import matplotlib.pyplot as plt
import numpy as np

from myMADDPG import MADDPG
from torch.utils.tensorboard import SummaryWriter
from toy_env import ToyEnv
import math
import torch

def get_env(env_name,render_mode=False):
    """create environment and get observation and action dimension of each agent in this environment"""
    new_env = None
    if env_name == 'toy_env':
        new_env = ToyEnv(render_mode=render_mode)


    new_env.reset()
    _dim_info = {}
    for agent_id in new_env.agents:
        _dim_info[agent_id] = []  # [obs_dim, act_dim]
        _dim_info[agent_id].append(new_env.observation_space(agent_id).shape[0]) #state_dim
        _dim_info[agent_id].append(new_env.action_space(agent_id).shape[0]) #action_dim

    return new_env, _dim_info

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='toy_env', help='name of the env',
                        choices=['toy_env'])
    parser.add_argument('--episode_num', type=int, default=180000,
                        help='total episode num during training procedure')
    parser.add_argument('--episode_length', type=int, default=30, help='steps per episode') #序列长度
    parser.add_argument('--learn_interval', type=int, default=3,
                        help='episodes interval between learning time')
    parser.add_argument('--random_steps', type=int, default=5e4,  #
                        help='random steps before the agent start to learn')
    parser.add_argument('--tau', type=float, default=0.02, help='soft update parameter')
    parser.add_argument('--gamma', type=float, default=0.95, help='discount factor')
    parser.add_argument('--buffer_capacity', type=int, default=int(1e6), help='capacity of replay buffer')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch-size of replay buffer')
    parser.add_argument('--actor_lr', type=float, default=0.001, help='learning rate of actor')
    parser.add_argument('--critic_lr', type=float, default=0.001, help='learning rate of critic')
    parser.add_argument('--action_bound', type=float, default=2*math.pi, help='upper bound of action')
    parser.add_argument('--noise_std', type=float, default=0.1*2*math.pi, help='std of noise')
    args = parser.parse_args()
    np.random.seed(0)
    torch.manual_seed(0)
    # create folder to save result
    env_dir = os.path.join('./results', args.env_name)
    if not os.path.exists(env_dir):
        os.makedirs(env_dir)
    total_files = len([file for file in os.listdir(env_dir)])
    result_dir = os.path.join(env_dir, f'{total_files + 1}')
    os.makedirs(result_dir)

    env, dim_info = get_env(args.env_name) 
    args.episode_length = env.episode_length
    maddpg = MADDPG(dim_info, args.buffer_capacity, args.batch_size, args.actor_lr, args.critic_lr,
                    result_dir)
    writer = SummaryWriter(result_dir)

    step = 0  # global step counter
    agent_num = env.num_agents
    # reward of each episode of each agent
    win_list = [] #一般胜
    win1_list = [] #大奖励
    red0_hp_l = []
    red1_hp_l = []
    red2_hp_l = []
    episode_rewards = {agent_id: np.zeros(args.episode_num) for agent_id in env.agents}
    for episode in range(args.episode_num):
        obs, infos = env.reset() #改1
        obs = env.Normalization(obs) #状态归一
        agent_reward = {agent_id: 0 for agent_id in env.agents}  # agent reward of the current episode
        #while env.agents:  # interact with the env for an episode #zh-cn:与环境交互一个episode
        #for _ in range(args.episode_length): #改3
        done = {agent_id: False for agent_id in env.agents}
        while not any(done.values()):  # interact with the env for an episode
            step += 1
            if step < args.random_steps: #此时action为(-1,1)
                action_nor = env.sample()# 也可以{agent_id: env.action_space(agent_id).sample() for agent_id in env.agents}
            else:
                action_nor = maddpg.select_action(obs)
            # 加噪音
            action = {agent_id: [np.clip(a[0]*args.action_bound + np.random.normal(0, args.noise_std,), 
                            -args.action_bound, args.action_bound)]
                            for agent_id, a in action_nor.items()}
            ## 单个值就行？ 还是得列表？ 都是列表的形式
            #print('action:',action_nor)
            #print('action:',action)
            next_obs, reward, terminations, truncations, info = env.step(action)
            next_obs = env.Normalization(next_obs) #状态归一
            
            done ={agent_id: terminations[agent_id] or truncations[agent_id] for agent_id in env.agents}

            #每个时间步惩罚和结算奖励  
            for agent_id, r in reward.items():
                #reward[agent_id] -= 0.01 # 暂时取消
                #r = abs(reward[agent_id])
                if info['win1'] == True: #只有结算时有 大win
                    reward[agent_id] +=  100#3*r  #100
                    if info[agent_id] > 1e-3: 
                        reward[agent_id] +=  100  #存活奖励
                        reward[agent_id] +=  info[agent_id]*100  #生命值奖励
                elif info['win2'] == True:  #小win
                    reward[agent_id] += 30#r #30
                # elif info['lose1'] == True:
                #     reward[agent_id] -= 3*r
                # elif info['lose2'] == True:
                #     reward[agent_id] -= r
            # env.render()
            maddpg.add(obs, action_nor, reward, next_obs, done) # 增加一步的经验

            for agent_id, r in reward.items():  # update reward
                agent_reward[agent_id] += r
                
            obs = next_obs

        # episode finishes
        if step >= args.random_steps and episode % args.learn_interval == 0:  # learn every few steps
            maddpg.learn(args.batch_size, args.gamma)
            #for _ in range(args.learn_interval):
            maddpg.update_target(args.tau)

        win_list.append(1 if info["win"] else 0) 
        win1_list.append(1 if info["win1"] else 0)
        if info["win1"]:
            red0_hp_l.append(info["Red-0"])
            red1_hp_l.append(info["Red-1"])
            red2_hp_l.append(info["Red-2"])

        for agent_id, r in agent_reward.items():  # record reward
            episode_rewards[agent_id][episode] = r
            writer.add_scalar(f'agent_{agent_id}_reward', r, episode)
        
        if (episode + 1) % 100 == 0:  # print info every 100 episodes
            message = f'episode {episode + 1}, '
            sum_reward = 0
            for agent_id, r in agent_reward.items():  # record reward
                message += f'{agent_id}: {r:.2f}; '
                sum_reward += r
            message += f'sum reward: {sum_reward:.2f}; '
            win_rate = np.mean(win_list[-100:]) #一般胜率
            win1_rate = np.mean(win1_list[-100:])
            writer.add_scalar('win_rate', win_rate, episode)
            writer.add_scalar('win1_rate', win1_rate, episode) #大胜利
            if len(red0_hp_l) > 100:
                red0_hp = np.mean(red0_hp_l[-100:])
                red1_hp = np.mean(red1_hp_l[-100:])
                red2_hp = np.mean(red2_hp_l[-100:])
                writer.add_scalar('red0_hp', red0_hp, episode)
                writer.add_scalar('red1_hp', red1_hp, episode)
                writer.add_scalar('red2_hp', red2_hp, episode)
                # 创建一个字典来存储多个标量数据 不好横向比较 暂时弃用
                # tag_scalar_dict = {
                # 'red0_hp': red0_hp,
                # 'red1_hp': red1_hp,
                # 'red2_hp': red2_hp
                # }
                # # 使用 add_scalars 方法将多个标量数据添加到同一个图表中
                # writer.add_scalars('red_hp', tag_scalar_dict, episode)


            message += f'red_hp:{info["Red-0"]:.2f},{info["Red-1"]:.2f},{info["Red-2"]:.2f};'
            message += f"hp:{info['Blue-0']:.2f},{info['Blue-1']:.2f},{info['Blue-2']:.2f}; "
            message += f'win rate: {win_rate:.2f}; '
            message += f'win1 rate: {win1_rate:.2f}; '
            print(message)

        


    maddpg.save(episode_rewards)  # save model


    def get_running_reward(arr: np.ndarray, window=100):
        """calculate the running reward, i.e. average of last `window` elements from rewards"""
        running_reward = np.zeros_like(arr)
        for i in range(window - 1):
            running_reward[i] = np.mean(arr[:i + 1])
        for i in range(window - 1, len(arr)):
            running_reward[i] = np.mean(arr[i - window + 1:i + 1])
        return running_reward


    # training finishes, plot reward
    fig, ax = plt.subplots()
    x = range(1, args.episode_num + 1)
    for agent_id, reward in episode_rewards.items():
        ax.plot(x, reward, label=agent_id)
        ax.plot(x, get_running_reward(reward))
    ax.legend()
    ax.set_xlabel('episode')
    ax.set_ylabel('reward')
    title = f'training result of maddpg solve {args.env_name}'
    ax.set_title(title)
    plt.savefig(os.path.join(result_dir, title))