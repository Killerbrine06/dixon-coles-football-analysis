import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Redefine the CGLS function for self-contained execution
def CGLS(X, Y, tol=1e-6, max_iter=None):
    n, p = X.shape
    if max_iter is None:
        max_iter = p + 10 

    beta = np.zeros(p)
    r = Y - X @ beta         
    s = X.T @ r              
    p_dir = s.copy()
    
    error_history = [np.linalg.norm(s)]
    
    for k in range(max_iter):
        q = X @ p_dir
        s_norm_sq = np.dot(s, s)
        q_norm_sq = np.dot(q, q)
        
        if q_norm_sq == 0:  
            break
            
        alpha = s_norm_sq / q_norm_sq
        beta = beta + alpha * p_dir
        r = r - alpha * q
        s_new = X.T @ r
        
        current_error = np.linalg.norm(s_new)
        error_history.append(current_error)
        
        if current_error < tol:
            break
            
        gamma = np.dot(s_new, s_new) / s_norm_sq
        p_dir = s_new + gamma * p_dir
        s = s_new
        
    return beta, error_history

# 2. Load the predictive features dataset
file_path = 'PSG_UCL_2023_24_Predictive_xG_Features.csv'
df = pd.read_csv(file_path)

# 3. Construct the Feature Matrix X and Target Vector Y
# We extract our selected features: Is_Home, Rest_Days, Form_Poss_3M, Form_Sh_3M, Form_SoT_3M
feature_cols = ['Is_Home', 'Rest_Days', 'Form_Poss_3M', 'Form_Sh_3M', 'Form_SoT_3M']

X_raw = df[feature_cols].values
Y_raw = df['GF'].values # Target: Goals For

# Add a column of 1s at the front of X to serve as our Intercept (Beta_0)
X_all = np.c_[np.ones(X_raw.shape[0]), X_raw]

# 4. Strict Train/Test Split to avoid Data Leakage
# Train data: Matches 1 to 10 (Indices 0 to 9)
X_train = X_all[0:8]
Y_train = Y_raw[0:8]

# Test data: Match 11 (Index 10 - Dortmund vs PSG, Semi-Final Leg 1)
X_match_11 = X_all[8:9] 
Y_match_11_actual = Y_raw[8]

# 5. Compute the Beta vector using CGLS
beta_solution, errors = CGLS(X_train, Y_train)

# 6. Make the Prediction for Match #11
# Prediction formula: Y_hat = X * Beta
predicted_goals = X_match_11 @ beta_solution

# 7. Output Results to Console
print("=" * 50)
print("             CGLS PREDICTION REPORT             ")
print("=" * 50)
print(f"Match #11 Context: Away vs {df['Opponent'].iloc[10]} (Date: {df['Date'].iloc[10]})")
print(f"Pre-Match Form: {df['Form_Sh_3M'].iloc[10]} Shots/Game, {df['Form_SoT_3M'].iloc[10]} SoT/Game")
print("-" * 50)
print(f"Calculated Intercept (Beta_0): {beta_solution[0]:.4f}")
print(f"Feature Weights (Beta_1 to Beta_5):")
for col, b in zip(feature_cols, beta_solution[1:]):
    print(f"  -> {col}: {b:.4f}")
print("-" * 50)
print(f"PREDICTED GOALS FOR MATCH #11 : {predicted_goals[0]:.2f}")
print(f"ACTUAL GOALS SCORED           : {Y_match_11_actual}")
print("=" * 50)

# # 8. Plot the Conjugate Gradient Residual Errors
# plt.figure(figsize=(8, 4))
# plt.plot(errors, marker='o', linestyle='-', color='indigo', linewidth=2)
# plt.title("CGLS Numerical Optimization: Residual Matrix Error")
# plt.xlabel("Iteration Step (K)")
# plt.ylabel("Normal Equation Residual Norm (||s||)")
# plt.yscale('log') # Log scale to clearly see convergence
# plt.grid(True, which="both", linestyle='--', alpha=0.5)
# plt.tight_layout()
# plt.show()