import os
from tqdm import tqdm
from utils import ModelTune, storeBestModel
param_grid = [(4, 2),\
        (4, 3),\
        (4, 4),\
        (9, 2),\
        (9, 3),\
        (9, 4)]
agent_list = ['dqn', 'a2c', 'ppo', 'sac']
agent_dict = {}
for (p,l) in tqdm(param_grid,desc='env'):
  for agent_name in tqdm(agent_list,desc=f'agent_{p}_{l}'):
    best_reward, best_params, agent = ModelTune(p, l, agent_name)
    storeBestModel(p, l, agent, agent_name)
    agent_dict[(p,l,agent_name)] = (best_params,agent)
    print(f"Best agent {agent_name} for p={p}, l={l} yielded")
import json
file_path = '/results/agent_dict.json'
with open(file_path, 'w') as file:
    json.dump(agent_dict, file)