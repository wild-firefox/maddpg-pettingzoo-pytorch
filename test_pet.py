from toy_env import PettingZoo_ToyEnv

env = PettingZoo_ToyEnv()
state , info = env.reset()
print('state:', state)
print('info:', info)
for agent_id in env.agents:
    print('agent_id:', agent_id)
print('agents',env.num_agents)
# print('state_space:', env.observation_space) # Box(0.0, 1.0, (12,), float32)
# print('action_space:', env.action_space) #Discrete(2)
print('state_dim:', env.observation_space("Red-0").shape[0])
print('action_dim:', env.action_space("Red-0").shape[0])
action = {agent: env.action_space(agent).sample() for agent in env.agents}
print('action:', action)

