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
  if n==0:
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

# AppendSpine recieves four parameters:
# list of particles with borning time and position (BBM_0), 
# the final time (T),
# an array [t,x0,y0] (b_part) (borned particle),
# the branching rate (g). 
# Appends into BBM_0 the list with positions and borning times 
# of particles borned along the spine borned at [t,x0,y0]

def AppendSpine(BBM_0, T, b_part, g):
  t = b_part[0]
  x0 = b_part[1]
  y0 = b_part[2]
  n = random.poisson(T-t)
  if n>0:
    #Borning times
    B = random.uniform(0,T-t,n)
    B.sort()
    B = B+t

    #Lifetimes
    l = np.empty(n)
    l[1:n] = B[1:n] - B[0:n-1]
    l[0] = B[0] - t
    
    #Position
    for i in range(n):
      x1 = x0 + random.normal(0,np.sqrt(l[i]))
      y1 = y0 + random.normal(0,np.sqrt(l[i]))
      
      #Poisson Thinning
      aux = random.uniform(0,1)
      if aux < g(x1,y1):
        BBM_0.append([B[i], x1, y1])
      x0 = x1
      y0 = y1
    
    #Final living particle
    x1 = x0 + random.normal(0, np.sqrt(T-B[n-1]))
    y1 = y0 + random.normal(0, np.sqrt(T-B[n-1]))
    BBM_0.append([T, x1, y1])
  
  else:
    x1 = x0 + random.normal(0, np.sqrt(T-t))
    y1 = y0 + random.normal(0, np.sqrt(T-t))
    BBM_0.append([T, x1, y1])

# BBM recieves four parameters:
# list of particles with borning time and position (BBM_0), 
# the final time (T),
# a small tolerance (eps),
# the branching rate (g). 
# Returns: a realization of the g-BBM

def g_BBM(BBM_0, T, eps, g):
  B = []
  
  while len(BBM_0) > 0:
    b_part = BBM_0.pop()
    if b_part[0] < T-eps:
      AppendSpine(BBM_0, T , b_part, g)
    
    else:
      B.append(b_part)

  return B