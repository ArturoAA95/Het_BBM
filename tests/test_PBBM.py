from BBM import InitializeBBM
from BBM import Pg_BBM
from BBM import PlotBBM
from BBM import PlotBBM_CH
from BranchingRates import g_2

T = 22
N = 18
eps = 10**(-15)
A = Pg_BBM([], T, N, eps, g_2)
for i in range(N):
    PlotBBM_CH(A[i], str(i), 26)

#for i in range(N):
#    PlotBBM(A[i])