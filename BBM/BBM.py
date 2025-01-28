import numpy as np
from numpy import random

# InitializeBBM receives:
# empty list (BBM_0), 
# the time (T),
# and the branching rate (g).
# Returns: list with positions and borning times 
# of particles borned along the spine 

def InitializeBBM(BBM_0, T, g):
  #Borning times: Poisson process 
  n = random.poisson(T)
  if n:
    BBM_0.append([T, random.normal(0, np.sqrt(T)),
                  random.normal(0, np.sqrt(T))])
    return BBM_0
  B = random.uniform(0, T, n)
  B.sort()

  #Lifetimes
  l = np.empty(n)
  l[1:n] = B[1:n] - B[0:n-1]
  l[0] = B[0]
  
  #Position
  x0 = 0
  y0 = 0
  for i in range(n):
    x1 = x0 + np.random.normal(0, np.sqrt(l[i]))
    y1 = y0 + np.random.normal(0, np.sqrt(l[i]))
    #Poisson Thinning
    aux = np.random.uniform(0, 1)
    if aux < g(x1, y1):
      BBM_0.append([B[i], x1, y1])
    x0 = x1
    y0 = y1
  
  #Final living particle
  x1 = x0 + np.random.normal(0, np.sqrt(T-B[n-1]))
  y1 = y0 + np.random.normal(0, np.sqrt(T-B[n-1]))
  BBM_0.append([T, x1, y1])
  
  return BBM_0

