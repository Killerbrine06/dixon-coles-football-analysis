import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# # 1. Generate synthetic data points for the example
# np.random.seed(42)
# x_all = np.linspace(0, 10, 15)
# # True underlying function: y = 2x^2 - 3x + 10 plus some noise
# y_all = 2 * x_all**3 - 3 * x_all + 10 + np.random.normal(0, 15, len(x_all))

# # 2. Pick one point to be the "out-of-sample" / real point
# # Let's take the 11th point (index 10)
# idx_real = 10
# x_real = x_all[idx_real]
# y_real = y_all[idx_real]

# # The rest are used for finding the approximation
# x_train = np.delete(x_all, idx_real)
# y_train = np.delete(y_all, idx_real)

file_path = 'PSG_UCL_2023_24_Merged_Stats.csv'
df = pd.read_csv(file_path)

NB_OF_GAMES = 10
COL_OF_INTEREST = 'SoT%'
x_train = [i for i in range(1, NB_OF_GAMES + 1)]
y_train = df[COL_OF_INTEREST].head(NB_OF_GAMES)

# 3. Best approximation: Least Squares Polynomial
degree = 3 # degree of the polynomial base
# np.polyfit inherently uses least squares optimization
coefficients = np.polyfit(x_train, y_train, degree)
polynomial = np.poly1d(coefficients)

# 4. Calculate the approximated value at the 'real' point's x coordinate
x_real = NB_OF_GAMES + 1
y_approx = polynomial(x_real)

# 5. Plotting the results
x_curve = np.linspace(min(x_train) - 1, max(x_train) + 1, 200)
y_curve = polynomial(x_curve)

y_real = df[COL_OF_INTEREST][NB_OF_GAMES]

plt.figure(figsize=(10, 6))

# Plot the points used for approximation
plt.scatter(x_train, y_train, color='blue', label='Data Points (Included)', s=50)

# Plot the least squares polynomial curve
plt.plot(x_curve, y_curve, color='orange', linewidth=2, label=f'Least Squares Polynomial (Degree {degree})')

# Plot the real point that was excluded
plt.scatter([x_real], [y_real], color='red', marker='*', s=200, label='Real Point (Excluded)', zorder=5)

# Plot the approximated point on the curve
plt.scatter([x_real], [y_approx], color='green', marker='X', s=150, label='Approximated Point', zorder=5)

# Draw a dashed line connecting the real point and the approximation to visualize the error
plt.vlines(x=x_real, ymin=min(y_real, y_approx), ymax=max(y_real, y_approx), colors='gray', linestyles='dashed', label='Prediction Error')

plt.title('Least Squares Polynomial Approximation')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# Save the plot
plt.savefig('ls_poli3_sot.png')
plt.close()