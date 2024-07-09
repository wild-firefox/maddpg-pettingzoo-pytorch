import pickle

# # 假设你有一个名为 'data.pkl' 的文件
# with open('results\\toy_env\\24\\rewards.pkl', 'rb') as file:
#     data = pickle.load(file)

# # 现在 'data' 变量包含了从 .pkl 文件中恢复的对象
# print(data['rewards']['Red-0'][-10:])
import time
import numpy as np
#随机种子
np.random.seed(0)
from toy_env import ToyEnv#,PettingZoo_ToyEnv
from pettingzoo.utils import parallel_to_aec, wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn
env = ToyEnv()
#env = parallel_to_aec(env) #报错
action_bound =2*np.pi
state , info = env.reset()

blue_win = 0
for _ in range(200):
    obs, infos = env.reset()
    done = {agent_id: False for agent_id in env.agents}
    while not any(done.values()):
        action = env.sample()#{agent: env.action_space(agent).sample() for agent in env.agents}
        # action = {agent_id: [0]
        #                 for agent_id, a in action.items()}
        action = {agent_id: [np.clip(a[0]*action_bound + np.random.normal(0,0.1*action_bound,), 
                    -action_bound, action_bound)]
                    for agent_id, a in action.items()}
        #print('action:',action)
        next_state, reward, terminations, truncations, info = env.step(action)
        done = {agent_id: terminations[agent_id] or truncations[agent_id] for agent_id in env.agents}
        
        # # 打印红蓝方的hp值
        message = f'step: { env.step_}, '
        message += f'Red-0 : {info["Red-0"]:.2f}, Red-1 : {info["Red-1"]:.2f}, Red-2 : {info["Red-2"]:.2f}, '
        message += f'Blue-0 : {info["Blue-0"]:.2f}, Blue-1 : {info["Blue-1"]:.2f}, Blue-2 : {info["Blue-2"]:.2f}, '
        # message += f'Red-0_xy : {info["Red-0_xy"]}, Red-1_xy : {info["Red-1_xy"]}, Red-2_xy : {info["Red-2_xy"]}, '
        # message += f'Blue-0_xy : {info["Blue-0_xy"]}, Blue-1_xy : {info["Blue-1_xy"]}, Blue-2_xy : {info["Blue-2_xy"]}, '
        # message += f'Red_0_fuel : {info["Red-0_fuel"]:.2f}, Red_1_fuel : {info["Red-1_fuel"]:.2f}, Red_2_fuel : {info["Red-2_fuel"]:.2f}, '
        # message += f'Blue_0_fuel : {info["Blue-0_fuel"]:.2f}, Blue_1_fuel : {info["Blue-1_fuel"]:.2f}, Blue_2_fuel : {info["Blue-2_fuel"]:.2f}, '
        message += f'lose : {info["lose"]},win : {info["win"]},win1 : {info["win1"]},win2 : {info["win2"]},'
        #print(message)
        blue_win += 1 if info["lose"] else 0
print(f'win_rate:{blue_win/200:.2f}')