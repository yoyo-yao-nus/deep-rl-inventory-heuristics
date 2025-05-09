import random, collections, copy
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.distributions import Categorical
import torch.nn.functional as F
from utils import Policy, Value, QNetwork

def cal_hurl_target(hu_func,value_net,buf,lmbda,gamma,n_step,q_flag=False):
    next_state_n = buf[-1][-1]
    nextstates = torch.tensor(np.array([traj[-1] for traj in buf]),dtype=torch.float,device=device)
    h_next = hu_func(nextstates)
    rewards = torch.tensor(np.array([traj[2] for traj in buf]),dtype=torch.float,device=device)
    reshaped_rewards = rewards + (1 - lmbda.item()) * gamma * h_next


    # Calculate n-step return with modified discount
    modified_gamma = lmbda * gamma
    cum_r = sum((modified_gamma ** t) * r for t, r in enumerate(reshaped_rewards))
    cum_r = torch.tensor(cum_r,dtype=torch.float32,device=device)

    with torch.no_grad():
      if q_flag:
        v_next = value_net(torch.tensor(next_state_n, dtype=torch.float32, device=device)).max().item()
      else:
        v_next = value_net(torch.tensor(next_state_n, dtype=torch.float32, device=device))
    hu_next = torch.tensor(hu_func(torch.tensor([next_state_n])),dtype=torch.float32,device=device)[0]
    target = cum_r + (modified_gamma ** n_step) * v_next + \
            (gamma ** n_step - modified_gamma ** n_step) * hu_next
    return torch.tensor(target,dtype=torch.float32,device=device)


def heuristic(obs_batch):
    global S, q_max
    I = obs_batch[:, 0]
    pipeline = obs_batch[:, 1:].sum(dim=1)

    mean_demand = torch.tensor(5.0, dtype=torch.float32, device=obs_batch.device)
    zero = torch.tensor(0.0, dtype=torch.float32, device=obs_batch.device)

    next_state_0 = torch.clamp(I + obs_batch[:, 1] - mean_demand, min=0, max=q_max)
    action = S - (I + pipeline)
    action = torch.clamp(action, min=0, max=q_max)

    global env
    cost = next_state_0 + env.p * torch.max(zero, mean_demand - I - obs_batch[:, 1])
    rewards = -cost
    return rewards


class HuAC_agent:
    def __init__(self, obs_dim, act_dim, n_step, gamma, lr_a=1e-3, lr_c=1e-3, lmbda=0, lmbda_schedule_steps=1000):
        self.n, self.g = n_step, gamma
        self.actor = Policy(obs_dim, act_dim).to(device)
        self.critic = Value(obs_dim).to(device)
        self.opt_a = optim.Adam(self.actor.parameters(), lr=lr_a)
        self.opt_c = optim.Adam(self.critic.parameters(), lr=lr_c)
        self.lmbda_init = torch.tensor(lmbda,dtype=torch.float32,device=device)
        self.current_lmbda = self.lmbda_init
        self.lmbda_schedule_steps = lmbda_schedule_steps
        self.step_count = 0

    def act(self, state):
        state = torch.tensor(state, dtype=torch.float, device=device).unsqueeze(0)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def _update_lambda(self):
        scale = torch.tensor(5.0,dtype=torch.float32,device=device)
        self.current_lmbda = self.lmbda_init + (1 - self.lmbda_init) * (1 - torch.exp(-scale * self.step_count/self.lmbda_schedule_steps))
        self.step_count += 1

    def update(self, buf):
        self._update_lambda()
        curr_state = buf[0][0]
        next_state_n = buf[-1][-1]
        target = cal_hurl_target(heuristic,self.critic,buf,self.current_lmbda,self.g,self.n)
        v_curr = self.critic(torch.tensor(curr_state,dtype=torch.float32,device=device))
        TD_error = target - v_curr
        critic_loss = TD_error.pow(2)/2

        states = torch.tensor(np.array([traj[0] for traj in buf]),dtype=torch.float).to(device)
        actions = torch.tensor(np.array([traj[1] for traj in buf])).to(device).unsqueeze(1)
        log_probs = torch.log(self.actor(states).gather(1, actions)+1e-8)
        actor_loss = torch.mean(-log_probs * TD_error.detach())

        self.opt_a.zero_grad()
        self.opt_c.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.opt_a.step()
        self.opt_c.step()
        return actor_loss.item() + critic_loss.item()

class HuDQN_agent(object):
  "Agent for the Deep Q Learning"
  "Your code should include the training process(update parameters) and action selection"
  def __init__(self, obs_dim, act_dim, n_step, gamma, lr=1e-3, lmbda=0, lmbda_schedule_steps=1000):
        self.n, self.g = n_step, gamma
        self.q = QNetwork(obs_dim, act_dim).to(device)
        self.q_tgt = copy.deepcopy(self.q).eval()
        self.opt = optim.Adam(self.q.parameters(), lr=lr)
        self.act_dim = act_dim
        self.eps, self.eps_end, self.eps_dec = 1.0, 0.05, 500
        self.lmbda_init = torch.tensor(lmbda,dtype=torch.float32,device=device)
        self.current_lmbda = self.lmbda_init
        self.lmbda_schedule_steps = lmbda_schedule_steps
        self.step_count = 0

  def act(self, obs):
        if random.random() < self._epsilon():
            return random.randrange(self.act_dim)
        with torch.no_grad():
            q = self.q(torch.tensor(obs,dtype=torch.float32,device=device))
        return int(q.argmax().item())

  def _update_lambda(self):
        scale = torch.tensor(5.0,dtype=torch.float32,device=device)
        self.current_lmbda = self.lmbda_init + (1 - self.lmbda_init) * (1 - torch.exp(-scale * self.step_count/self.lmbda_schedule_steps))
        self.step_count += 1

  def _epsilon(self):
        self.eps = max(self.eps_end, self.eps - (1 - self.eps_end) / self.eps_dec)
        return self.eps

  def learn(self, buf):
        self._update_lambda()
        curr_state = buf[0][0]
        curr_action = buf[0][1]
        q_curr = self.q(torch.tensor(curr_state,dtype=torch.float32).to(device))[curr_action]
        target = cal_hurl_target(heuristic,self.q_tgt,buf,self.current_lmbda,self.g,self.n,q_flag=True)
        TD_error = target - q_curr
        loss = TD_error.pow(2)/2
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        return loss.item()

  def update_tgt(self):
        self.q_tgt.load_state_dict(self.q.state_dict())

class HuPPOAgent:
    def __init__(self, obs_dim, act_dim, n_step, gamma, lmbda=0.95, lr_a=3e-4, lr_c=3e-3, hu_lmbda=0, lmbda_schedule_steps=1000):
        self.n, self.g, self.l = n_step, gamma, lmbda
        self.actor = Policy(obs_dim, act_dim).to(device)
        self.critic = Value(obs_dim).to(device)
        self.opt_a = optim.Adam(self.actor.parameters(), lr=lr_a)
        self.opt_c = optim.Adam(self.critic.parameters(), lr=lr_c)
        self.lr_sch_a = optim.lr_scheduler.StepLR(self.opt_a, step_size=500, gamma=0.1)
        self.lr_sch_c = optim.lr_scheduler.StepLR(self.opt_c, step_size=500, gamma=0.1)
        self.K_epochs = 10
        self.eps_clip = 0.2
        self.hu_lmbda_init = torch.tensor(hu_lmbda,dtype=torch.float32,device=device)
        self.hu_lmbda = self.hu_lmbda_init
        self.lmbda_schedule_steps = lmbda_schedule_steps
        self.step_count = 0



    def act(self, obs):
          logits = self.actor(torch.tensor(obs, dtype=torch.float, device=device))
          dist = torch.distributions.Categorical(logits=logits)
          return int(dist.sample().item())

    def _update_lambda(self):
        scale = torch.tensor(5.0,dtype=torch.float32,device=device)
        self.hu_lmbda = self.hu_lmbda_init + (1 - self.hu_lmbda_init) * (1 - torch.exp(-scale * self.step_count/self.lmbda_schedule_steps))
        self.step_count += 1

    def update(self, buf):
        self._update_lambda()
        states = torch.tensor([traj[0] for traj in buf],dtype=torch.float32).to(device)
        actions = torch.tensor([traj[1] for traj in buf]).to(device).unsqueeze(1)
        rewards = torch.tensor([traj[2] for traj in buf],dtype=torch.float32).to(device)
        nextstates = torch.tensor([traj[-1] for traj in buf],dtype=torch.float32).to(device)
        hu_nexts = heuristic(nextstates)
        new_rewards = rewards + (1 - self.hu_lmbda) * self.g * hu_nexts
        new_gamma = self.hu_lmbda * self.g
        td_target = new_rewards + new_gamma * self.critic(nextstates)
        td_delta = td_target - self.critic(states)
        advantage = self.compute_advantage(td_delta.cpu()).to(device)
        old_log_probs = torch.log(self.actor(states).gather(1,actions)+1e-8).detach()

        for _ in range(self.K_epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions)+1e-8)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps_clip,1 + self.eps_clip) * advantage
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
            self.opt_a.zero_grad()
            self.opt_c.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.opt_a.step(); self.lr_sch_a.step()
            self.opt_c.step(); self.lr_sch_c.step()
        return (actor_loss + critic_loss).item()


    def compute_advantage(self, td_delta):
        td_delta = td_delta.detach().numpy()
        advantage_list = []
        advantage = 0.0
        for delta in td_delta[::-1]:
            advantage = self.g * self.l * advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()
        return torch.tensor(advantage_list, dtype=torch.float)

class HuSACAgent:
    def __init__(self, obs_dim, act_dim, n_step, gamma, target_entropy=-1, tau=0.005, lr_a=3e-4, lr_c=3e-3, lr_alpha=1e-2, lmbda=0, lmbda_schedule_steps=1000):
        self.n, self.g = n_step, gamma
        self.target_entropy = target_entropy
        self.tau = tau
        self.actor = Policy(obs_dim, act_dim).to(device)
        self.critic1 = QNetwork(obs_dim, act_dim).to(device)
        self.critic2 = QNetwork(obs_dim, act_dim).to(device)
        self.critic1_tgt = copy.deepcopy(self.critic1).eval()
        self.critic2_tgt = copy.deepcopy(self.critic2).eval()
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float, device=device)
        self.log_alpha.requires_grad = True

        self.opt_a = optim.Adam(self.actor.parameters(), lr=lr_a)
        self.opt_c1 = optim.Adam(self.critic1.parameters(), lr=lr_c)
        self.opt_c2 = optim.Adam(self.critic2.parameters(), lr=lr_c)
        self.log_alpha_optimizer = optim.Adam([self.log_alpha],lr=lr_alpha)

        self.lmbda_init = torch.tensor(lmbda,dtype=torch.float32,device=device)
        self.current_lmbda = self.lmbda_init
        self.lmbda_schedule_steps = lmbda_schedule_steps
        self.step_count = 0

    def act(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        logits = self.actor(obs)
        dist = torch.distributions.Categorical(logits=logits)
        return int(dist.sample().item())

    def _update_lambda(self):
        scale = torch.tensor(5.0,dtype=torch.float32,device=device)
        self.current_lmbda = self.lmbda_init + (1 - self.lmbda_init) * (1 - torch.exp(-scale * self.step_count/self.lmbda_schedule_steps))
        self.step_count += 1

    def calc_target(self, rewards, next_state_n):
        next_state_n_tensor = torch.tensor(next_state_n,dtype=torch.float32,device=device)
        next_probs = self.actor(next_state_n_tensor)
        next_log_probs = torch.log(next_probs + 1e-8)
        entropy = -torch.sum(next_probs * next_log_probs)
        q1_value = self.critic1_tgt(next_state_n_tensor)
        q2_value = self.critic2_tgt(next_state_n_tensor)
        min_qvalue = next_probs * torch.min(q1_value, q2_value)
        next_value = min_qvalue + self.log_alpha.exp() * entropy
        modified_gamma = self.g * self.current_lmbda
        discounted_rewards = torch.sum(torch.tensor([modified_gamma**i*reward.item() for i, reward in enumerate(rewards[:-1])],device=device))
        hu_next = torch.tensor(heuristic(torch.tensor([next_state_n])),dtype=torch.float32,device=device)[0]
        td_target = discounted_rewards + modified_gamma**self.n * next_value + \
         (self.g**self.n - modified_gamma**self.n) * hu_next
        return td_target

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(),net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self, buf):
        self._update_lambda()
        states = torch.tensor([traj[0] for traj in buf],dtype=torch.float32).to(device)
        actions = torch.tensor([traj[1] for traj in buf]).to(device).unsqueeze(1)
        rewards = torch.tensor([traj[2] for traj in buf],dtype=torch.float32).to(device)
        next_states = torch.tensor([traj[-1] for traj in buf],dtype=torch.float32).to(device)
        next_state_n = buf[-1][-1]
        next_state_n_tensor = torch.tensor(next_state_n,dtype=torch.float32,device=device)
        h_next = heuristic(next_states)
        rewards = torch.tensor(np.array([traj[2] for traj in buf]),dtype=torch.float,device=device)
        rewards = rewards + (1 - self.current_lmbda.item()) * self.g * h_next
        next_probs = self.actor(next_state_n_tensor)
        td_target = self.calc_target(rewards, next_state_n)
        critic_1_q_values = self.critic1(states)
        critic_1_loss = torch.mean(
            F.mse_loss(critic_1_q_values, td_target.detach()))
        critic_2_q_values = self.critic2(states)
        critic_2_loss = torch.mean(
            F.mse_loss(critic_2_q_values, td_target.detach()))
        self.opt_c1.zero_grad()
        critic_1_loss.backward()
        self.opt_c1.step()
        self.opt_c2.zero_grad()
        critic_2_loss.backward()
        self.opt_c2.step()

        probs = self.actor(states)
        log_probs = torch.log(probs + 1e-8)
        entropy = -torch.sum(probs * log_probs, dim=1, keepdim=True)
        q1_value = self.critic1(states)
        q2_value = self.critic2(states)
        min_qvalue = probs * torch.min(q1_value, q2_value)
        actor_loss = -torch.mean(self.log_alpha.exp() * entropy + min_qvalue)
        self.opt_a.zero_grad()
        actor_loss.backward()
        self.opt_a.step()

        alpha_loss = torch.mean(
            (entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic1, self.critic1_tgt)
        self.soft_update(self.critic2, self.critic2_tgt)
        return (actor_loss + critic_1_loss + critic_2_loss).item()