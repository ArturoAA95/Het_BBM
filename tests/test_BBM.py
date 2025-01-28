from BBM import InitializeBBM
from BBM import g_BBM
from BranchingRates import g_1

T = 3
B = InitializeBBM([], T, g_1)
eps = 10**(-15)
A = g_BBM(B, T, eps, g_1)
print(A)
