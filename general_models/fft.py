import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Define the spacing and generate the x-axis points
file_path = 'PSG_UCL_2023_24_Merged_Stats.csv'
df = pd.read_csv(file_path)

NB_OF_GAMES = 10
COL_OF_INTEREST = 'SoT%'
x = [i for i in range(1, NB_OF_GAMES + 1)]
y = df[COL_OF_INTEREST].head(NB_OF_GAMES)

# 2. Perform the Fast Fourier Transform (FFT)
# This maps our spatial data y into the frequency domain
Y_fft = np.fft.fft(y)

# 3. Perform the Inverse Fast Fourier Transform (IFFT)
# This maps the frequency data back into the spatial domain, reconstructing the signal
y_approx = np.fft.ifft(Y_fft)

# 4. Plot the results
plt.figure(figsize=(10, 6))

# Plot the original data points
plt.plot(x, y, 'ro', label='Original Data Points', markersize=6, zorder=5)

# Plot the reconstructed approximation
# Note: IFFT often returns numbers with infinitesimal imaginary components (e.g., + 0.00000001j) 
# due to floating-point arithmetic. We use .real to plot purely the real numbers.
plt.plot(x, y_approx.real, 'b-', label='Reconstructed Approximation (IFFT)', linewidth=2, alpha=0.8)

plt.title('FFT and Inverse FFT Reconstruction ($h=1$)')
plt.xlabel('X (spacing $h=1$)')
plt.ylabel('Y')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# Save the plot
plt.savefig('fft_reconstruction.png')
plt.close()