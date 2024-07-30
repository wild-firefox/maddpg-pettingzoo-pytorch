from toy_env import ToyEnv

import torch
env = ToyEnv()
state , info = env.reset()
print('state:', state)  
print('info:', info)

print('agents',env.agents)


#action = {agent: env.action_space(agent).sample() for agent in env.agents}
action = env.sample()
print('action:', action)
#print('agent_index:',list(action.keys()).index('Red-0'))
observations, rewards, terminations, truncations, infos = env.step(action)
print('observations:', observations)

print('rewards:', rewards)
print('terminations:', terminations)
print('truncations:', truncations)
print('infos:', infos)
state_dim = env.observation_space("Red-0").shape[0]
print('state_dim:', state_dim)
action_dim = env.action_space("Red-0").shape[0]
print('action_dim:', action_dim)

# 为observations和action中的每个张量增加一个维度
for i, v in observations.items():
    observations[i] = torch.unsqueeze(torch.tensor(v), dim=0)
for i, v in action.items():
    action[i] = torch.unsqueeze(torch.tensor(v), dim=0)
print('ob_value:',list(observations.values()))
print('act_value:',list(action.values()))
print('cat_value:',torch.cat(list(observations.values())+list(action.values()),dim =1))

# #while env.agents:  # interact with the env for an episode
done = {agent_id: False for agent_id in env.agents}
#for _ in range(200):
while not any(done.values()):
    action = env.sample()#{agent: env.action_space(agent).sample() for agent in env.agents}
    next_state, reward, terminations, truncations, info = env.step(action)
    done = {agent_id: terminations[agent_id] or truncations[agent_id] for agent_id in env.agents}
    # 打印红蓝方的hp值
    message = f'episode { env.step_}, '
    message += f'Red-0 : {info["Red-0"]}, Red-1 : {info["Red-1"]}, Red-2 : {info["Red-2"]}, '
    message += f'Blue-0 : {info["Blue-0"]}, Blue-1 : {info["Blue-1"]}, Blue-2 : {info["Blue-2"]}, '

    print(message)


from myAgent import Agent
import torch

agent = Agent(12,1,36,0.01,0.01)
obs = torch.tensor([1,2,3,4,5,6,7,8,9,10,11,12],dtype=torch.float32)
print(agent.action(obs))
print(agent.target_action(obs))

from myMADDPG import MADDPG
from toy_env import ToyEnv

def get_env(env_name):
    """create environment and get observation and action dimension of each agent in this environment"""
    new_env = None
    if env_name == 'toy_env':
        new_env = ToyEnv()


    new_env.reset()
    _dim_info = {}
    for agent_id in new_env.agents:
        _dim_info[agent_id] = []  # [obs_dim, act_dim]
        _dim_info[agent_id].append(new_env.observation_space(agent_id).shape[0])
        _dim_info[agent_id].append(new_env.action_space(agent_id).shape[0])

    return new_env, _dim_info
dim_info = get_env('toy_env')[1]
maddpg = MADDPG(dim_info, 1000, 1024, 0.01, 0.01,'test')

observations, rewards, terminations, truncations, infos = env.step(action)
print(maddpg.select_action(observations))

print(env.Normalization(observations))
print(env.episode_length)

global_obs_act_dim = sum(sum(val) for val in dim_info.values())
print(global_obs_act_dim)