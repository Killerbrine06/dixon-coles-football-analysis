import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# # 1. Define the set of points 
# # (Note: For splines, the X values must be strictly increasing)
# x = np.array([0, 1, 2, 3, 4, 5])
# y = np.array([3, 1, 4, 1, 5, 9])

file_path = 'PSG_UCL_2023_24_Merged_Stats.csv'
df = pd.read_csv(file_path)

NB_OF_GAMES = 10
COL_OF_INTEREST = 'SoT%'
x = [i for i in range(1, NB_OF_GAMES + 1)]
y = df[COL_OF_INTEREST].head(NB_OF_GAMES)

# 2. Create the Cubic Spline interpolation
# bc_type='natural' automatically enforces the boundary condition: 
# f''(x[0]) = 0 and f''(x[-1]) = 0
cs = CubicSpline(x, y, bc_type='natural')

# 3. Generate a highly sampled array of X values to draw a smooth curve
x_curve = np.linspace(min(x), max(x) + 1, 200)
y_curve = cs(x_curve)


x_real = NB_OF_GAMES + 1
y_approx = cs(x_real)
y_real = df[COL_OF_INTEREST][NB_OF_GAMES]

# 4. Plot the data points and the resulting spline
plt.figure(figsize=(10, 6))

# Plot the original points
plt.scatter(x, y, color='red', s=100, label='Data Points', zorder=5)

# Plot the spline interpolation
plt.plot(x_curve, y_curve, color='blue', linewidth=2, label='Natural Cubic Spline ($C^2$)')
plt.scatter([x_real], [y_real], color='red', marker='*', s=200, label='Real Point (Excluded)', zorder=5)

# Plot the approximated point on the curve
plt.scatter([x_real], [y_approx], color='green', marker='X', s=150, label='Approximated Point', zorder=5)
plt.vlines(x=x_real, ymin=min(y_real, y_approx), ymax=max(y_real, y_approx), colors='gray', linestyles='dashed', label='Prediction Error')

# Formatting the plot
plt.title('Natural Cubic Spline Interpolation')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# Save the plot
plt.savefig('natural_cubic_spline.png')
plt.close()