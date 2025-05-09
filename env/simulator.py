import numpy as np
import random, collections, copy

class Lostsale(object):
  "Please provide your code here, your code should at least include the init function and step function which returns the next state and reward"
  "Your code should be adaptive the buffer size, here the buffer size is assumed to be 10"
  def __init__(self, buffer_size = 100, gamma = 0.99, p=4, l=2):
    self.state_dim = l+1
    self.n_action = 11
    self.discount = gamma
    self.p = p # variable
    self.buffer_size = buffer_size
    self.lead = l # variable
    self.ini_state = np.zeros(l+1)
    self.state=None
    self.buffer = Replaybuffer(capacity=buffer_size, n_step=1, gamma=gamma)
    self.reset()

  def step(self, state, action, demand):
    next_state = np.zeros(self.lead+1)
    next_state[0] = max(state[0] + state[1] - demand, 0)
    for i in range(self.lead-1):
      next_state[i+1] = state[i+2]
    next_state[-1] = action #next_state[0] = action
    cost = next_state[0] + self.p*max(0, demand - state[0] - state[1])
    rewards = -cost
    return next_state.copy(), rewards


  ### BEGIN HERE
  def reset(self):
        self.state = self.ini_state.copy()
        return self.state.copy()

  def sample_demand(self, lam=5.0):
        return np.random.poisson(lam)

  def env_step(self, action, demand=None):
        if demand is None:
            demand = self.sample_demand()
        transition = [self.state.copy(),action]
        next_state, reward = self.step(self.state, action, demand)
        self.state = next_state
        transition.extend([reward, next_state.copy()])
        self.buffer.push(transition)
        return next_state.copy(), reward

class Replaybuffer(object):
  "The Replaybuffer is designed for you to store the history sample paths"
  "You can interact with the Replaybuffer to update your parameters"
  def __init__(self, capacity=100_000, n_step=1, gamma=0.99):
        self.buf = collections.deque(maxlen=capacity)
        self.n    = n_step
        self.g    = gamma
  def __repr__(self):
        return self.buf.__repr__()

  def push(self, traj):
        self.buf.append(traj)

  def sample(self, batch_size):
        start = random.randint(0, len(self.buf) - batch_size)
        batch = list(self.buf)[start:start+batch_size]
        return batch

  def __len__(self):
        return len(self.buf)


  ### END HERE