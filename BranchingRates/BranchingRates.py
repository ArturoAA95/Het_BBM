import numpy as np
import math

#Rate functions for the braching Brownian motion

def g_1(x,y):
  return (1/4)*np.sin(x*2*(np.pi))*np.sin(y*2*(np.pi))+(3/4)

def g_2(x,y):
  xf=x-math.floor(x)
  yf=y-math.floor(y)
  if(xf==0):
    return 0
  m=yf/xf
  if xf==0:
    return 1
  if m<(1/3):
    return .2
  if m>(3):
    return .2
  else:
    return 1
  
def g_3(x,y):
  return np.cos(2*x*np.pi)+2