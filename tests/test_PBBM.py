from BBM import InitializeBBM
from BBM import Pg_BBM
from BBM import PlotBBM
from BranchingRates import g_2

T = 15
N = 5
eps = 10**(-15)
A = Pg_BBM([], T, N, eps, g_2)
for b in A:
    PlotBBM(b)

#for i in range(N):
#    PlotBBM(A[i])