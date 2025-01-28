import numpy as np
import math

#Rate functions for the braching Brownian motion

def g_1(x,y):
  return (1/4)*np.sin(x*2*(np.pi))*np.sin(y*2*(np.pi))+(3/4)

def g_2(x,y):
  xf=x/2-math.floor(x/2)
  yf=y/2-math.floor(y/2)
  m=yf/xf
  if m<(1/3):
    return .2
  if m>(3):
    return .2
  else:
    return 1