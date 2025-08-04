from BBM import InitializeBBM
from BBM import Pg_BBM
from BBM import PlotBBM
from BBM import PlotBBM_CH
from BranchingRates import g_2
from BranchingRates import g_0
from BBM import g_BBM_killing
import numpy as np

T = 60
wd = int(T*np.sqrt(2)) + 5
N = 12
eps = 10**(-15)
#A = Pg_BBM([], T, N, eps, g_0)
A = g_BBM_killing([], T, N, eps, g_0)
for i in range(N):
    PlotBBM_CH(A[i], str(i), wd, (i+1)*(T/N))

#for i in range(N):
#    PlotBBM(A[i])