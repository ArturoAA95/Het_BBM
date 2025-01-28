from BBM import InitializeBBM
from BBM import g_BBM
from BranchingRates import g_1

B = InitializeBBM([], 5, g_1)
print(B)
eps = 10**(-15)
A = g_BBM(B, 5, eps, g_1)
print(A)
