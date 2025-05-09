import numpy as np
import torch, torch.nn as nn, torch.optim as optim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.distributions import Categorical
import torch.nn.functional as F
from env.simulator import Lostsale

def mlp(in_dim, out_dim, hidden=(128, 128), act=nn.ReLU, out_act=None):
    layers, dim = [], in_dim
    for h in hidden:
        layers += [nn.Linear(dim, h), act()]
        dim = h
    layers += [nn.Linear(dim, out_dim)]
    if out_act: layers.append(out_act())
    return nn.Sequential(*layers)


class QNetwork(nn.Module):
  "The deep network for training your model"
  def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = mlp(obs_dim, act_dim)
  def forward(self, x):
        return self.net(x)

class Policy(nn.Module):
  "Policy network if you are using the policy-based algorithms"
  ### BEGIN HERE
  def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = mlp(obs_dim, act_dim)
        self.softmax = nn.Softmax(dim=-1)
  def forward(self, x):
    x = self.net(x)
    return self.softmax(x)
  ### END HERE

class Value(nn.Module):
  def __init__(self, obs_dim):
        super().__init__()
        self.net = mlp(obs_dim, 1)
  def forward(self, x):
        return self.net(x)

class ContinuousPolicy(nn.Module):
  def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = mlp(obs_dim, act_dim)

def policy_entropy(obs,actor):
    logits = actor(torch.tensor(obs, dtype=torch.float, device=device).unsqueeze(0))
    dist = torch.distributions.Categorical(logits=logits)
    return dist.entropy().item()

def evaluate_policy(env, func, sim_steps=10000, n_repl=5, **kwargs):
    obs  = env.reset()
    ret, discount = 0.0, 1.0
    for _ in range(sim_steps):
        a = func(obs, **kwargs)
        obs, r = env.env_step(a)
        ret += discount * r
        discount *= env.discount
    return ret


from heuristics.heuristics import Basestock,Cappedbasestock,Constantorder,Myopic1
import itertools
def grid_search_basestock(p=4, l=2, S_grid=range(5, 41, 2)):
    env = make_env(p, l)
    best_S, best_ret = None, -np.inf
    for S in S_grid:
        ret = evaluate_policy(env, Basestock, sim_steps=20_000, S=S)
        if ret > best_ret:
            best_S, best_ret = S, ret
    return best_S, best_ret


def grid_search_capped(p=4, l=2,\
            S_grid=range(5, 41, 5),\
            q_grid=(5, 10, 15, 20)):
    env = make_env(p, l)
    best_pair, best_ret = None, -np.inf
    for S, q in itertools.product(S_grid, q_grid):
        ret = evaluate_policy(env, Cappedbasestock,
                              sim_steps=20_000,
                              S=S, q_max=q)
        if ret > best_ret:
            best_pair, best_ret = (S, q), ret
    return best_pair, best_ret


def grid_search_constant(p=4, l=2, r_grid=range(2, 21, 2)):
    env = make_env(p, l)
    best_r, best_ret = None, -np.inf
    for r in r_grid:
        ret = evaluate_policy(env, Constantorder,
                    sim_steps=20_000,
                    r=r)
        if ret > best_ret:
            best_r, best_ret = r, ret
    return best_r, best_ret


def eval_myopic(p=4, l=2):
    env = make_env(p, l)
    ret = evaluate_policy(env,\
        Myopic1,sim_steps=20_000,\
        p=p, h=1, lamda=5.0, l=l)
    return ret



from tqdm import tqdm
def train_dqn(env, agent, epochs=200, num_steps=100, buffer_sample_size=10, tgt_update_step=10, learn_step=5):
    value_curve, loss_curve, epoch_loss_curve = [], [], []
    b = buffer_sample_size
    c = tgt_update_step
    n = learn_step

    # Check agent's model(s) on GPU
    assert next(agent.q.parameters()).is_cuda, "Agent's Q network is not on GPU!"
    assert next(agent.q_tgt.parameters()).is_cuda, "Agent's target Q network is not on GPU!"

    for ep in range(epochs):
        obs = env.reset()
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32, device=device)
        else:
            obs = obs.to(device)
        assert obs.device.type == "cuda", "Observation tensor is not on GPU!"

        ep_ret, discount = 0.0, 1.0
        epoch_loss = []

        for step in range(num_steps):
            loss = None

            if step % c == 0:
                agent.update_tgt()

            action = agent.act(obs)
            next_obs, reward = env.env_step(action)

            if isinstance(next_obs, np.ndarray):
                next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
            else:
                next_obs = next_obs.to(device)
            assert next_obs.device.type == "cuda", "Next observation tensor is not on GPU!"

            ep_ret += discount * reward
            discount *= env.discount

            if step >= b and step % n == 0:
                batch = env.buffer.sample(b)
                loss = agent.learn(batch)
                epoch_loss.append(loss)

            obs = next_obs
            if loss is not None:
                loss_curve.append(loss)

        value_curve.append(ep_ret)
        ep_loss_scalar = np.mean(epoch_loss)
        epoch_loss_curve.append(ep_loss_scalar)
        # print(f"EP {ep:03d} | cost {-ep_ret:9.1f} | loss {ep_loss_scalar:.4f}")

    return value_curve, loss_curve, epoch_loss_curve


def train_actor_critic(env, agent, epochs=200, num_steps=100, buffer_sample_size=10, learn_step=5):
    value_curve, loss_curve, epoch_loss_curve = [], [], []
    entropy_value = []
    b = buffer_sample_size
    n = learn_step

    # Check model is on GPU
    assert next(agent.actor.parameters()).is_cuda, "Agent's actor is not on GPU!"
    # assert next(agent.critic.parameters()).is_cuda, "Agent's critic is not on GPU!"

    for ep in range(epochs):
        obs = env.reset()
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32, device=device)
        else:
            obs = obs.to(device)
        assert obs.device.type == "cuda", "Observation tensor is not on GPU!"

        ep_ret, discount = 0.0, 1.0
        epoch_loss = []
        epoch_entropy = []

        for step in range(num_steps):
            loss = None

            # Act and ensure device consistency
            action = agent.act(obs)
            next_obs, reward = env.env_step(action)

            if isinstance(next_obs, np.ndarray):
                next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
            else:
                next_obs = next_obs.to(device)
            assert next_obs.device.type == "cuda", "Next observation tensor is not on GPU!"

            ep_ret += discount * reward
            discount *= env.discount

            # Learning step
            if step >= b and step % n == 0:
                sample = env.buffer.sample(b)
                loss = agent.update(sample)
                epoch_loss.append(loss)

            obs = next_obs
            if loss is not None:
                loss_curve.append(loss)

            entropy = policy_entropy(obs, agent.actor)
            epoch_entropy.append(entropy)

        ep_entropy_scalar = np.mean(epoch_entropy)
        entropy_value.append(ep_entropy_scalar)
        ep_loss_scalar = np.mean(epoch_loss)
        epoch_loss_curve.append(ep_loss_scalar)
        value_curve.append(ep_ret)

        # print(f"EP {ep:03d} | cost {-ep_ret:9.1f} | loss {ep_loss_scalar:.4f}")

    return value_curve, loss_curve, epoch_loss_curve, entropy_value


def make_env(p,l):
    return Lostsale(p=p, l=l)

def base_policy(obs, S=20, max_order=10, lead=2):
    I = obs[0]
    pipeline = sum(obs[1:])
    action = max(0, S - (I + pipeline))
    return min(action, max_order)
def train_base(env, policy_func, epochs=200, buffers_per_epoch=100):
    value_curve = []
    for ep in range(epochs):
        obs = env.reset()
        ep_ret, discount = 0.0, 1.0
        for _ in range(buffers_per_epoch * env.buffer_size):
            #discount = 1.0
            a = policy_func(obs)
            obs, r = env.env_step(a)
            ep_ret += discount * r
            discount *= env.discount
        value_curve.append(ep_ret)
        print(f"[BASE] EP {ep:03d} | cost {-ep_ret:9.1f}")
    return value_curve

import os
def storeBestModel(p, l, agent, agent_name):
    model_dir = f"/content/drive/MyDrive/Colab Notebooks/models/p_{p}_l_{l}"  # Create a directory for each (p, l) combination
    os.makedirs(model_dir, exist_ok=True)
    if 'dqn' in agent_name:
      torch.save({
        'model_state_dict': agent.q.state_dict(),
        }, os.path.join(model_dir, f"{agent_name}_model.pth"))
    else:
       torch.save({
        'actor_state_dict': agent.actor.state_dict(),
        }, os.path.join(model_dir, f"{agent_name}_model.pth"))

from configs import hyperparameters
from agents.rlagents import DQN_agent,AC_agent,PPOAgent,SACAgent
from hurl.hurlagents import HuDQN_agent,HuAC_agent,HuPPOAgent,HuSACAgent
def ModelTune(price, lead, agent_name):
    global env
    env = make_env(price, lead)
    best_reward = float('-inf')
    best_params = None
    best_agent = None

    epochs = 300
    num_steps = 100
    buffer_sample_size = 10
    learn_step = 5
    if agent_name == 'dqn':
        hyperparam = hyperparameters['dqn']
        tgt_update_step = 10
        for lr in tqdm(hyperparam['lr'],desc='lr'):
            agent = DQN_agent(obs_dim=env.state_dim, act_dim=env.n_action, n_step=env.buffer_size, gamma=env.discount, lr=lr)
            v, l, epoch_l = train_dqn(env, agent, epochs=epochs, num_steps=num_steps, buffer_sample_size=buffer_sample_size, tgt_update_step=tgt_update_step, learn_step=learn_step)
            if np.mean(v[-100:]) > best_reward:
                best_reward = np.mean(v[-100:])
                best_params = {'lr': lr}
                best_agent = agent
    elif agent_name == 'a2c':
        hyperparam = hyperparameters['a2c']
        for lr_a in tqdm(hyperparam['lr_a'],desc='lr_a'):
            for lr_c in tqdm(hyperparam['lr_c'],desc='lr_c'):
                agent = AC_agent(obs_dim=env.lead+1, act_dim=env.n_action, n_step=env.buffer_size,
                                 gamma=env.discount, lr_a=lr_a, lr_c=lr_c)
                v, l, epoch_l, ent = train_actor_critic(env, agent,
                                      epochs=epochs,
                                      num_steps=num_steps,
                                      buffer_sample_size=buffer_sample_size,
                                      learn_step=learn_step)
                if np.mean(v[-100:]) > best_reward:
                    best_reward = np.mean(v[-100:])
                    best_params = {'lr_a': lr_a, 'lr_c': lr_c}
                    best_agent = agent
    elif agent_name == 'ppo':
        hyperparam = hyperparameters['ppo']
        for lr_a in tqdm(hyperparam['lr_a'],desc='lr_a'):
            for lr_c in tqdm(hyperparam['lr_c'],desc='lr_c'):
                for lmbda in hyperparam['lmbda']:
                    agent = PPOAgent(obs_dim=env.lead+1, act_dim=env.n_action, n_step=env.buffer_size,
                                     gamma=env.discount, lmbda=lmbda, lr_a=lr_a, lr_c=lr_c)
                    v, l, epoch_l, ent = train_actor_critic(env, agent,
                                      epochs=epochs,
                                      num_steps=num_steps,
                                      buffer_sample_size=buffer_sample_size,
                                      learn_step=learn_step)
                    if np.mean(v[-100:]) > best_reward:
                        best_reward = np.mean(v[-100:])
                        best_params = {'lr_a': lr_a, 'lr_c': lr_c, 'lmbda': lmbda}
                        best_agent = agent
    elif agent_name == 'sac':
        hyperparam = hyperparameters['sac']
        for lr_a in tqdm(hyperparam['lr_a'],desc='lr_a'):
            for lr_c in tqdm(hyperparam['lr_c'],desc='lr_c'):
                for lr_alpha in hyperparam['lr_alpha']:
                    for target_entropy in hyperparam['target_entropy']:
                        for tau in hyperparam['tau']:
                            agent = SACAgent(obs_dim=env.lead+1, act_dim=env.n_action,
                                             n_step=env.buffer_size,
                                             gamma=env.discount,
                                             target_entropy=target_entropy,
                                             tau=tau)
                            v, l, epoch_l, ent = train_actor_critic(env, agent,
                                      epochs=epochs,
                                      num_steps=num_steps,
                                      buffer_sample_size=buffer_sample_size,
                                      learn_step=learn_step)
                            if np.mean(v[-100:]) > best_reward:
                                best_reward = np.mean(v[-100:])
                                best_params = {'lr_a': lr_a, 'lr_c': lr_c, 'lr_alpha': lr_alpha, 'target_entropy': target_entropy, 'tau': tau}
                                best_agent = agent

    elif agent_name == 'hudqn':
        hyperparam = hyperparameters['dqn']
        tgt_update_step = 10
        for lr in tqdm(hyperparam['lr'],desc='lr'):
            agent = HuDQN_agent(obs_dim=env.state_dim, act_dim=env.n_action, n_step=env.buffer_size, gamma=env.discount, lr=lr)
            v, l, epoch_l = train_dqn(env, agent, epochs=epochs, num_steps=num_steps, buffer_sample_size=buffer_sample_size, tgt_update_step=tgt_update_step, learn_step=learn_step)
            if np.mean(v[-100:]) > best_reward:
                best_reward = np.mean(v[-100:])
                best_params = {'lr': lr}
                best_agent = agent
    elif agent_name == 'hua2c':
        hyperparam = hyperparameters['a2c']
        for lr_a in tqdm(hyperparam['lr_a'],desc='lr_a'):
            for lr_c in tqdm(hyperparam['lr_c'],desc='lr_c'):
                agent = HuAC_agent(obs_dim=env.lead+1, act_dim=env.n_action, n_step=env.buffer_size,
                                 gamma=env.discount, lr_a=lr_a, lr_c=lr_c)
                v, l, epoch_l, ent = train_actor_critic(env, agent,
                                      epochs=epochs,
                                      num_steps=num_steps,
                                      buffer_sample_size=buffer_sample_size,
                                      learn_step=learn_step)
                if np.mean(v[-100:]) > best_reward:
                    best_reward = np.mean(v[-100:])
                    best_params = {'lr_a': lr_a, 'lr_c': lr_c}
                    best_agent = agent
    elif agent_name == 'huppo':
        hyperparam = hyperparameters['ppo']
        for lr_a in tqdm(hyperparam['lr_a'],desc='lr_a'):
            for lr_c in tqdm(hyperparam['lr_c'],desc='lr_c'):
                for lmbda in hyperparam['lmbda']:
                    agent = HuPPOAgent(obs_dim=env.lead+1, act_dim=env.n_action, n_step=env.buffer_size,
                                     gamma=env.discount, lmbda=lmbda, lr_a=lr_a, lr_c=lr_c)
                    v, l, epoch_l, ent = train_actor_critic(env, agent,
                                      epochs=epochs,
                                      num_steps=num_steps,
                                      buffer_sample_size=buffer_sample_size,
                                      learn_step=learn_step)
                    if np.mean(v[-100:]) > best_reward:
                        best_reward = np.mean(v[-100:])
                        best_params = {'lr_a': lr_a, 'lr_c': lr_c, 'lmbda': lmbda}
                        best_agent = agent
    elif agent_name == 'husac':
        hyperparam = hyperparameters['sac']
        for lr_a in tqdm(hyperparam['lr_a'],desc='lr_a'):
            for lr_c in tqdm(hyperparam['lr_c'],desc='lr_c'):
                for lr_alpha in hyperparam['lr_alpha']:
                    for target_entropy in hyperparam['target_entropy']:
                        for tau in hyperparam['tau']:
                            agent = HuSACAgent(obs_dim=env.lead+1, act_dim=env.n_action,
                                             n_step=env.buffer_size,
                                             gamma=env.discount,
                                             target_entropy=target_entropy,
                                             tau=tau)
                            v, l, epoch_l, ent = train_actor_critic(env, agent,
                                      epochs=epochs,
                                      num_steps=num_steps,
                                      buffer_sample_size=buffer_sample_size,
                                      learn_step=learn_step)
                            if np.mean(v[-100:]) > best_reward:
                                best_reward = np.mean(v[-100:])
                                best_params = {'lr_a': lr_a, 'lr_c': lr_c, 'lr_alpha': lr_alpha, 'target_entropy': target_entropy, 'tau': tau}
                                best_agent = agent

    else:
        raise ValueError("Invalid agent name")
    return best_reward, best_params, best_agent