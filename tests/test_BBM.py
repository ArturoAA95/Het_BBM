from BBM import InitializeBBM
from BBM import g_BBM
from BBM import PlotBBM
from BranchingRates import g_2

T = 15
B = InitializeBBM([], T, g_2)
eps = 10**(-15)
A = g_BBM(B, T, eps, g_2)

PlotBBM(A)

