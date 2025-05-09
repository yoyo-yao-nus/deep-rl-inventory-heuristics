from tqdm import tqdm
import os
from configs import hu_dict
from utils import ModelTune, storeBestModel
param_grid = [(4, 2),\
        (4, 3),\
        (4, 4),\
        (9, 2),\
        (9, 3),\
        (9, 4)]
agent_list = ['huppo', 'husac']
huagent_dict = {}
for (p,l) in tqdm(param_grid,desc='env'):
  global S, q_max
  (S, q_max) = hu_dict[(p,l)] # CappedBaseStock dictionary
  for agent_name in tqdm(agent_list,desc=f'agent_{p}_{l}'):
    # filename = f'/content/drive/MyDrive/Colab Notebooks/models/p_{p}_l_{l}/{agent_name}_model.pth'
    # if os.path.exists(filename):
    #   print(f'Model {agent_name} for p={p} l={l} exists.')
    #   continue
    best_reward, best_params, agent = ModelTune(p, l, agent_name)
    storeBestModel(p, l, agent, agent_name)
    huagent_dict[(p,l,agent_name)] = (best_params, agent)
    print(f"Best agent {agent_name} for p={p}, l={l} yielded")
import json
file_path = '/content/drive/MyDrive/Colab Notebooks/models/huagent_dict.json'
with open(file_path, 'w') as file:
    json.dump(huagent_dict, file)
