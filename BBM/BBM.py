import numpy as np
from numpy import random
import matplotlib.pyplot as plt # type: ignore
from scipy.spatial import ConvexHull # type: ignore

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

def AppendSpineBanana(BBM_0, T, b_part, g, r):
  t = b_part[0]
  x0 = b_part[1]
  y0 = b_part[2]
  if np.sqrt(x0**2 + y0**2) > r + (T-t)*np.sqrt(2)*(.4)-1:
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

# g_BBM recieves four parameters:
# list of particles with borning time and position (BBM_0), 
# the final time (T),
# a small tolerance (eps),
# the branching rate (g). 
# Returns: a realization of the g-BBM

def Recenter(BBM_0, x_mean, y_mean):
  if len(BBM_0) == 0:
    return BBM_0
  for i in range(len(BBM_0)):
    BBM_0[i][1] -= x_mean
    BBM_0[i][2] -= y_mean
  return BBM_0

def g_BBM(BBM_0, T, eps, g, r ):
  B = []
  while len(BBM_0) > 0:
    b_part = BBM_0.pop()
    if b_part[0] < T-eps:
      if b_part[0] > 10:
        AppendSpineBanana(BBM_0, T , b_part, g, r)
      else: AppendSpine(BBM_0, T , b_part, g)
    else:
      B.append(b_part)
  return B

#ws window size

def PlotBBM(BBM_0, name, ws, T):
  N = len(BBM_0)
  x = np.empty(N)
  y = np.empty(N)
  for i in range (N):
    x[i] = BBM_0[i][1]
    y[i] = BBM_0[i][2]

  fig, ax = plt.subplots(figsize=(6, 6))
  ax.scatter(x, y, s=1)
  ax.axis('equal')
  plt.title("T={0}".format(T))
  plt.xlim(-ws, ws)
  plt.ylim(-ws, ws)
  fig.savefig('{0}.png'.format(name))   # save the figure to file
  plt.close(fig)
  plt.show()


def DistanceSimplex(A, B):
    """
    Compute the perpendicular distance from the origin to the line formed by points A and B in 2D.
    A, B: numpy arrays or lists with shape (2,)
    """
    # Convert to numpy arrays if not already
    A = np.asarray(A)
    B = np.asarray(B)
    # Direction vector
    v = B - A
    # Numerator: |(B_x - A_x)*A_y - (B_y - A_y)*A_x|
    numerator = abs(v[0]*A[1] - v[1]*A[0])
    # Denominator: sqrt((B_x - A_x)^2 + (B_y - A_y)^2)
    denominator = np.linalg.norm(v)
    if denominator == 0:
        return np.linalg.norm(A)  # If A and B are the same point, return distance to origin
    return numerator / denominator

#ws window size

def PlotBBM_CH(BBM_0, name, ws, T):
  N = len(BBM_0)
  x = np.empty(N)
  y = np.empty(N)
  for i in range (N):
    x[i] = BBM_0[i][1]
    y[i] = BBM_0[i][2]

  points = np.column_stack((x, y))
  hull = ConvexHull(points)
  fig, ax = plt.subplots(figsize=(6, 6))
  ax.scatter(x, y, s=1)
  r = 1000
  for simplex in hull.simplices:
    #distance from the origin of the simplex
    x = points[simplex, 0]
    y = points[simplex, 1]
    r_aux = DistanceSimplex( [x[0], y[0]], [x[1], y[1]])
    if r_aux < r:
      r = r_aux
    #Plot
    ax.plot(points[simplex, 0], points[simplex, 1], 'r')
    
  if r - np.log(T+1)*3/(np.sqrt(2)*2)-1 > 0:
    r = r - np.log(T+1)*3/(np.sqrt(2)*2)-1
  
  circle1 = plt.Circle((0, 0), r, color='r', fill=False)
  ax.add_patch(circle1)
  ax.plot(points[hull.vertices, 0], points[hull.vertices, 1], 'o', mec='r', color='r')
  
  plt.title("T={0}".format(T))
  ax.axis('equal')
  plt.xlim(-ws, ws)
  plt.ylim(-ws, ws)
  fig.savefig('{0}.png'.format(name))   # save the figure to file
  plt.close(fig)
  plt.show()
  return r
# Pg_BBM recieves five parameters:
# list of particles with borning time and position (BBM_0), 
# the final time (T),
# number pictures (N)
# a small tolerance (eps),
# the branching rate (g). 
# Returns: a realization of the g-BBM

def Pg_BBM(BBM_0, T, N, eps, g):
  pics = []
  t = np.empty(N)
  A = InitializeBBM([], T/N, g)
  for i in range(N):
    t[i] = (i+1)*T/N
    B = g_BBM(A.copy(), t[i], eps, g)
    #Recenter the BBM
    if t[i] > 9 and t[i] < 15:
      B = Recenter(B)
    A = B.copy()
    #PlotBBM(A)
    pics.append(A)
  return pics

def g_BBM_killing(BBM_0, T, N, eps, g):
    pics = []
    t = np.empty(N)
    A = InitializeBBM([], T/N, g)
    r = 0
    for i in range(N):
        print(i)
        t[i] = (i+1)*T/N
        B = g_BBM(A.copy(), t[i], eps, g, r)
        #Recenter the BBM
        P, C = CH(B)
        #x_mean, y_mean = CenterCH(P, C)
        cent , r = ChebyshevCenter(P, C)
        print(r)
        print((r-np.sqrt(2)*t[i])*np.sqrt(2)*2/np.log(t[i]))
        B = Recenter(B, cent[0], cent[1])
        r, points, hull = RadiousCH(B)
        #print(hull.equations)
        if r - np.log(t[i]+1)*3/(np.sqrt(2)*2) - 1 > 0:
            r = r - np.log(t[i]+1)*3/(np.sqrt(2)*2) - 1
        else:
            r = 0

        A = B.copy()
        pics.append(A)
    return pics

def RadiousCH(BBM_0):
    N = len(BBM_0)
    BBM_0 = np.array(BBM_0)  # Convert to numpy array for slicing
    x = BBM_0[:, 1]
    y = BBM_0[:, 2]
    points = np.column_stack((x, y))
    hull = ConvexHull(points)
    r = 1000
    for simplex in hull.simplices:
        a = points[simplex, 0]
        b = points[simplex, 1]
        r_aux = DistanceSimplex([a[0], b[0]], [a[1], b[1]])
        if r_aux < r:
            r = r_aux
    return r, points, hull


def CH(BBM_0):
    N = len(BBM_0)
    BBM_0 = np.array(BBM_0)  # Convert to numpy array for slicing
    x = BBM_0[:, 1]
    y = BBM_0[:, 2]
    points = np.column_stack((x, y))
    hull = ConvexHull(points)
    return points, hull

def CenterCH(points, hull):
    x_mean = np.mean(points[hull.vertices, 0])
    y_mean = np.mean(points[hull.vertices, 1])
    return x_mean, y_mean


from scipy.spatial import HalfspaceIntersection
from scipy.optimize import linprog

def ChebyshevCenter(points, hull):
  halfspaces = np.array(hull.equations)
  #feasible_point = np.array([0, 0])
  #hs = HalfspaceIntersection(halfspaces, feasible_point)
  norm_vector = np.reshape(np.linalg.norm(halfspaces[:, :-1], axis=1),
    (halfspaces.shape[0], 1))
  c = np.zeros((halfspaces.shape[1],))
  c[-1] = -1
  A = np.hstack((halfspaces[:, :-1], norm_vector))
  b = - halfspaces[:, -1:]
  res = linprog(c, A_ub=A, b_ub=b, bounds=(None, None))
  x = res.x[:-1]
  y = res.x[-1]
  
  return x, y