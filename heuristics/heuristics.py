import itertools
def Basestock(obs, **kwargs):
  ### BEGIN HERE
  I = obs[0]
  pipeline = sum(obs[1:])
  S = kwargs.get('S', 0)
  action = max(0, S - (I + pipeline))
  ### END HERE
  return action

def Cappedbasestock(obs,**kwargs):
  ### BEGIN HERE
  I = obs[0]
  pipeline = sum(obs[1:])
  S = kwargs.get('S', 0)
  q_max = kwargs.get('q_max', None)
  action = max(0, S - (I + pipeline))
  if q_max is not None:
      action= min(action, q_max)
  ### END HERE
  return action

def Constantorder(obs,**kwargs):
  ### BEGIN HERE
  r = kwargs.get('r', 0)
  action = max(0, r)
  ### END HERE
  return action

def Myopic1(obs,**kwargs):
  ### BEGIN HERE
  I = obs[0]
  pipeline = sum(obs[1:])
  lam = kwargs.get('lambda', 5.0)
  l = kwargs.get('l', 2)
  q_max = kwargs.get('q_max', None)
  p = kwargs.get('p', 4)
  h = kwargs.get('h', 1)

  lam_L = lam * l # parameter of the sum of poisson distributions
  alpha = h / (p + h) # ordering cost c=0
  k = 0 ; cdf = 0.0
  from math import exp, factorial
  while True:
      pmf  = exp(-lam_L) * lam_L**k / factorial(k)
      cdf += pmf
      if 1 - cdf <= alpha: # P(I_t<0) = P(I_{t-1}+pipeline+z-D_L<0) = P(D_L>I_{t-1}+pipeline+z) = 1-P(D_L<=I_{t-1}+pipeline+z)
          break
      k += 1

  action = max(0, k - (I + pipeline))
  if q_max is not None:
      action = min(action, q_max)
  ### END HERE
  return action