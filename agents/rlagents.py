import random, collections, copy
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.distributions import Categorical
import torch.nn.functional as F
from utils import Policy, Value, QNetwork

class AC_agent:
    def __init__(self, obs_dim, act_dim, n_step, gamma, lr_a=1e-3, lr_c=1e-3):
        self.n, self.g = n_step, gamma
        self.actor = Policy(obs_dim, act_dim).to(device)
        self.critic = Value(obs_dim).to(device)
        self.opt_a = optim.Adam(self.actor.parameters(), lr=lr_a)
        self.opt_c = optim.Adam(self.critic.parameters(), lr=lr_c)
        self.lr_sch_a = optim.lr_scheduler.StepLR(self.opt_a, step_size=500, gamma=0.1)
        self.lr_sch_c = optim.lr_scheduler.StepLR(self.opt_c, step_size=500, gamma=0.1)

    def act(self, state):
        state = torch.tensor(state, dtype=torch.float, device=device).unsqueeze(0)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, buf):
        curr_state = buf[0][0]
        next_state_n = buf[-1][-1]
        rewards = [traj[2] for traj in buf]
        cum_r = sum((self.g ** p) * r for p, r in enumerate(rewards))
        v_curr = self.critic(torch.tensor(curr_state,dtype=torch.float32).to(device))
        with torch.no_grad():
          v_next = self.critic(torch.tensor(next_state_n,dtype=torch.float32).to(device))
        target = cum_r + (self.g ** self.n) * v_next
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
        self.opt_a.step(); self.lr_sch_a.step()
        self.opt_c.step(); self.lr_sch_c.step()
        return actor_loss.item() + critic_loss.item()

class DQN_agent(object):
  "Agent for the Deep Q Learning"
  "Your code should include the training process(update parameters) and action selection"
  def __init__(self, obs_dim, act_dim, n_step, gamma, lr=1e-3):
        self.n, self.g = n_step, gamma
        self.q = QNetwork(obs_dim, act_dim).to(device)
        self.q_tgt = copy.deepcopy(self.q).eval()
        self.opt = optim.Adam(self.q.parameters(), lr=lr)
        self.act_dim = act_dim
        self.eps, self.eps_end, self.eps_dec = 1.0, 0.05, 500 # decreasing exploration rate

  def act(self, obs):
        if random.random() < self._epsilon():
            return random.randrange(self.act_dim)
        with torch.no_grad():
            q = self.q(torch.tensor(obs,dtype=torch.float32,device=device))
        return int(q.argmax().item())

  def _epsilon(self):
        self.eps = max(self.eps_end, self.eps - (1 - self.eps_end) / self.eps_dec)
        return self.eps

  def learn(self, buf):
        curr_state = buf[0][0]
        curr_action = buf[0][1]
        next_state_n = buf[-1][-1]
        rewards = [traj[2] for traj in buf]
        cum_r = sum((self.g ** p) * r for p, r in enumerate(rewards))
        q_curr = self.q(torch.tensor(curr_state,dtype=torch.float32).to(device))[curr_action]
        with torch.no_grad():
            q_next = self.q_tgt(torch.tensor(next_state_n,dtype=torch.float32).to(device)).max().item()
        target = cum_r + (self.g ** self.n) * q_next
        TD_error = target - q_curr
        loss = TD_error.pow(2)/2
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        return loss.item()
  def update_tgt(self):
        self.q_tgt.load_state_dict(self.q.state_dict())
        

class PPOAgent:
    def __init__(self, obs_dim, act_dim, n_step, gamma, lmbda=0.95, lr_a=3e-4, lr_c=3e-3):
        self.n, self.g, self.l = n_step, gamma, lmbda
        self.actor = Policy(obs_dim, act_dim).to(device)
        self.critic = Value(obs_dim).to(device)
        self.opt_a = optim.Adam(self.actor.parameters(), lr=lr_a)
        self.opt_c = optim.Adam(self.critic.parameters(), lr=lr_c)
        self.lr_sch_a = optim.lr_scheduler.StepLR(self.opt_a, step_size=500, gamma=0.1)
        self.lr_sch_c = optim.lr_scheduler.StepLR(self.opt_c, step_size=500, gamma=0.1)
        self.K_epochs = 10
        self.eps_clip = 0.2

    def act(self, obs):
          obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
          logits = self.actor(obs)
          dist = torch.distributions.Categorical(logits=logits)
          return int(dist.sample().item())

    def update(self, buf):
        states = torch.tensor([traj[0] for traj in buf],dtype=torch.float32).to(device)
        actions = torch.tensor([traj[1] for traj in buf]).to(device).unsqueeze(1)
        rewards = torch.tensor([traj[2] for traj in buf],dtype=torch.float32).to(device)
        nextstates = torch.tensor([traj[-1] for traj in buf],dtype=torch.float32).to(device)

        td_target = rewards + self.g * self.critic(nextstates).squeeze(-1)
        td_delta = td_target - self.critic(states).squeeze(-1)
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
        advantage_list = []
        advantage = 0.0
        for delta in reversed(td_delta.squeeze().tolist()):
            advantage = self.g * self.l * advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()
        return torch.tensor(advantage_list, dtype=torch.float32, device=device)


class SACAgent:
    def __init__(self, obs_dim, act_dim, n_step, gamma, target_entropy, tau, lr_a=3e-4, lr_c=3e-3, lr_alpha=1e-2):
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
        self.lr_sch_a = optim.lr_scheduler.StepLR(self.opt_a, step_size=500, gamma=0.1)
        self.lr_sch_c1 = optim.lr_scheduler.StepLR(self.opt_c1, step_size=500, gamma=0.1)
        self.lr_sch_c2 = optim.lr_scheduler.StepLR(self.opt_c2, step_size=500, gamma=0.1)
        self.log_alpha_optimizer = optim.Adam([self.log_alpha],lr=lr_alpha)

    def act(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        logits = self.actor(obs)
        dist = torch.distributions.Categorical(logits=logits)
        return int(dist.sample().item())

    def calc_target(self, rewards, next_state_n):
        next_probs = self.actor(next_state_n)
        next_log_probs = torch.log(next_probs + 1e-8)
        entropy = -torch.sum(next_probs * next_log_probs)
        q1_value = self.critic1_tgt(next_state_n)
        q2_value = self.critic2_tgt(next_state_n)
        min_qvalue = next_probs * torch.min(q1_value, q2_value)
        next_value = min_qvalue + self.log_alpha.exp() * entropy
        discounted_rewards = torch.sum(torch.tensor([self.g**i*reward.item() for i, reward in enumerate(rewards[:-1])],device=device))
        td_target = discounted_rewards + self.g**self.n * next_value
        return td_target

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(),net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self, buf):
        states = torch.tensor([traj[0] for traj in buf],dtype=torch.float32).to(device)
        actions = torch.tensor([traj[1] for traj in buf]).to(device).unsqueeze(1)
        rewards = torch.tensor([traj[2] for traj in buf],dtype=torch.float32).to(device)
        next_states = torch.tensor([traj[-1] for traj in buf],dtype=torch.float32).to(device)
        next_state_n = next_states[-1]
        td_target = self.calc_target(rewards, next_state_n)
        critic_1_q_values = self.critic1(states)
        critic_1_loss = torch.mean(
            F.mse_loss(critic_1_q_values, td_target.detach()))
        critic_2_q_values = self.critic2(states)
        critic_2_loss = torch.mean(
            F.mse_loss(critic_2_q_values, td_target.detach()))
        self.opt_c1.zero_grad()
        critic_1_loss.backward()
        self.opt_c1.step(); self.lr_sch_c1.step()
        self.opt_c2.zero_grad()
        critic_2_loss.backward()
        self.opt_c2.step(); self.lr_sch_c2.step()

        probs = self.actor(states)
        log_probs = torch.log(probs + 1e-8)
        entropy = -torch.sum(probs * log_probs, dim=1, keepdim=True)
        q1_value = self.critic1(states)
        q2_value = self.critic2(states)
        min_qvalue = probs * torch.min(q1_value, q2_value)
        actor_loss = -torch.mean(self.log_alpha.exp() * entropy + min_qvalue)
        self.opt_a.zero_grad()
        actor_loss.backward()
        self.opt_a.step(); self.lr_sch_a.step()

        alpha_loss = torch.mean(
            (entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic1, self.critic1_tgt)
        self.soft_update(self.critic2, self.critic2_tgt)
        return (actor_loss + critic_1_loss + critic_2_loss).item()