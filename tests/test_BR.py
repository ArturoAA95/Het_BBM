import matplotlib.pyplot as plt # type: ignore
import numpy as np
from BranchingRates import g_1
from BranchingRates import g_2

# Define the window for the heatmap
x_min = 0
x_max = 2
y_min = 0
y_max = 2

# Create a grid of x and y values within the window
x = np.linspace(x_min, x_max, 100)
y = np.linspace(y_min, y_max, 100)
X, Y = np.meshgrid(x, y)

# Calculate the function values for each point in the grid
Z = np.empty((100,100))
for i in range(100):
  for j in range(100):
    Z[i,j] = g_2(X[i,j],Y[i,j])

# Create the heatmap
plt.imshow(Z, extent=[x_min, x_max, y_min, y_max], origin='lower', cmap='viridis')
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Heatmap of g_1(x,y)')
plt.show()
