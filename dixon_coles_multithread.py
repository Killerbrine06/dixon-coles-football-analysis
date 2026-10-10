"""
Master Dixon-Coles Pipeline with Walk-Forward Validation and Grid Search.
Calculates optimal time-decay, evaluates via Brier Score, and exports JSON.
Includes Math Fixes, Normalized Probabilities, L2 Scaling, and Warm-Starting.
"""

import os, datetime
import json
import concurrent.futures
import math
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from datetime import timedelta

# ==========================================
# 1. MATHEMATICAL CORE (NLL & PROBABILITIES)
# ==========================================

def compute_nll_active(active_params, t1_mapped, t2_mapped, gf, ga, home_status, weights, num_active):
    a = active_params[:num_active]
    b = active_params[num_active:2 * num_active]
    g = active_params[-2]
    rho = active_params[-1]

    g_t1 = np.where(home_status == 1, g, 0.0)
    g_t2 = np.where(home_status == 2, g, 0.0)

    lambda_val = np.exp(a[t1_mapped] + b[t2_mapped] + g_t1)
    mu_val = np.exp(a[t2_mapped] + b[t1_mapped] + g_t2)

    tau = np.ones(len(gf))
    idx_00 = (gf == 0) & (ga == 0)
    idx_10 = (gf == 1) & (ga == 0)
    idx_01 = (gf == 0) & (ga == 1)
    idx_11 = (gf == 1) & (ga == 1)

    # FIX 1: Proper Dixon-Coles tau terms (Swap fixed)
    tau[idx_00] = 1.0 - (lambda_val[idx_00] * mu_val[idx_00] * rho)
    tau[idx_10] = 1.0 + (mu_val[idx_10] * rho)      # Fixed: uses mu
    tau[idx_01] = 1.0 + (lambda_val[idx_01] * rho)  # Fixed: uses lambda
    tau[idx_11] = 1.0 - rho
    tau = np.clip(tau, 1e-10, None)

    log_likelihood = weights * (
        gf * np.log(lambda_val) - lambda_val +
        ga * np.log(mu_val) - mu_val +
        np.log(tau)
    )

    total_nll = -np.sum(log_likelihood)
    
    # FIX 2: Sample-Size-Invariant Regularization
    effective_n = max(1.0, np.sum(weights))
    collinearity_penalty = 10000.0 * (np.mean(a)**2)
    l2_penalty = (10.0 / effective_n) * (np.sum(a**2) + np.sum(b**2))

    return total_nll + collinearity_penalty + l2_penalty

def calculate_match_probabilities(alpha_1, beta_1, alpha_2, beta_2, gamma, rho, home_status, max_goals=12):
    """Calculates win/draw/loss probabilities for a single match."""
    gamma_1 = gamma if home_status == 1 else 1.0
    gamma_2 = gamma if home_status == 2 else 1.0
    
    lambda_val = alpha_1 * beta_2 * gamma_1
    mu_val = alpha_2 * beta_1 * gamma_2
    
    prob_t1_win = 0.0
    prob_draw = 0.0
    prob_t2_win = 0.0
    
    total_prob = 0.0
    matrix = np.zeros((max_goals, max_goals))
    
    for x in range(max_goals):
        for y in range(max_goals):
            p = (math.pow(lambda_val, x) * math.exp(-lambda_val) / math.factorial(x)) * \
                (math.pow(mu_val, y) * math.exp(-mu_val) / math.factorial(y))
            
            # Dixon-Coles low-score correction (Fixed indices)
            if x == 0 and y == 0: p *= max(0, 1 - lambda_val * mu_val * rho)
            elif x == 1 and y == 0: p *= max(0, 1 + mu_val * rho)
            elif x == 0 and y == 1: p *= max(0, 1 + lambda_val * rho)
            elif x == 1 and y == 1: p *= max(0, 1 - rho)
            
            matrix[x, y] = p
            total_prob += p
            
    # Normalize probabilities so they sum exactly to 1.0
    matrix /= total_prob
    
    for x in range(max_goals):
        for y in range(max_goals):
            if x > y: prob_t1_win += matrix[x, y]
            elif x == y: prob_draw += matrix[x, y]
            else: prob_t2_win += matrix[x, y]
            
    return prob_t1_win, prob_draw, prob_t2_win

def calculate_brier_score(prob_t1, prob_draw, prob_t2, actual_gf, actual_ga):
    """Calculates the Brier Score for a probability prediction."""
    if actual_gf > actual_ga: outcome = np.array([1, 0, 0])
    elif actual_gf == actual_ga: outcome = np.array([0, 1, 0])
    else: outcome = np.array([0, 0, 1])
        
    probabilities = np.array([prob_t1, prob_draw, prob_t2])
    return np.sum((probabilities - outcome) ** 2)

# ==========================================
# 2. PIPELINE ENGINES
# ==========================================

def solve_parameters(train_df, xi, num_teams, initial_params=None):
    active_teams = np.unique(np.concatenate((train_df['t1_idx'].values, train_df['t2_idx'].values)))
    num_active = len(active_teams)
    
    local_map = {global_idx: local_idx for local_idx, global_idx in enumerate(active_teams)}
    
    t1_mapped = np.array([local_map[idx] for idx in train_df['t1_idx'].values])
    t2_mapped = np.array([local_map[idx] for idx in train_df['t2_idx'].values])
    
    gf = train_df['GF'].values
    ga = train_df['GA'].values
    home_status = train_df['TeamHome'].values if 'TeamHome' in train_df.columns else np.ones(len(train_df), dtype=int)
    
    anchor_date = train_df['Date'].max()
    days_ago = (anchor_date - train_df['Date']).dt.days.values
    weights = np.exp(-xi * days_ago)

    if initial_params is None:
        params_0 = np.concatenate((
            np.zeros(num_active),  
            np.zeros(num_active),  
            [np.log(1.1)],        
            [0.0]                  
        ))
    else:
        # Map global previous params to local active params
        prev_a = initial_params[:num_teams]
        prev_b = initial_params[num_teams:2*num_teams]
        a_init = np.array([np.log(prev_a[idx]) for idx in active_teams])
        b_init = np.array([np.log(prev_b[idx]) for idx in active_teams])
        params_0 = np.concatenate((a_init, b_init, [np.log(initial_params[-2])], [initial_params[-1]]))

    bounds = [(-3.0, 3.0)] * (2 * num_active) + [(0.0, 0.5), (-0.2, 0.2)]

    result = minimize(
        compute_nll_active, params_0,
        args=(t1_mapped, t2_mapped, gf, ga, home_status, weights, num_active),
        method='L-BFGS-B', 
        bounds=bounds,
        options={'maxiter': 2000, 'ftol': 1e-4} # Slightly relaxed tolerance for speed
    )
    
    final_alphas = np.ones(num_teams)
    final_betas = np.ones(num_teams)
    
    optimized_alphas = np.exp(result.x[:num_active])
    optimized_betas = np.exp(result.x[num_active:2 * num_active])
    
    for local_idx, global_idx in enumerate(active_teams):
        final_alphas[global_idx] = optimized_alphas[local_idx]
        final_betas[global_idx] = optimized_betas[local_idx]
        
    final_gamma = np.exp(result.x[-2])
    final_rho = result.x[-1]

    return np.concatenate((final_alphas, final_betas, [final_gamma], [final_rho]))

def evaluate_single_xi(args):
    df, xi, num_teams, initial_days, step_days = args
    
    total_brier = 0
    predictions_made = 0
    start_date = df['Date'].min()
    end_date = df['Date'].max()
    current_train_end = start_date + timedelta(days=initial_days)
    
    # Warm starting state
    prev_params = None
    
    while current_train_end < end_date:
        train_mask = df['Date'] <= current_train_end
        test_mask = (df['Date'] > current_train_end) & (df['Date'] <= current_train_end + timedelta(days=step_days))
        
        train_df = df[train_mask]
        test_df = df[test_mask]
        
        if not test_df.empty and len(train_df) > 50:
            params = solve_parameters(train_df, xi, num_teams, initial_params=prev_params)
            prev_params = params # Save for warm start
            
            alphas = params[:num_teams]
            betas = params[num_teams:2 * num_teams]
            gamma, rho = params[-2], params[-1]
            
            for _, row in test_df.iterrows():
                p_t1, p_draw, p_t2 = calculate_match_probabilities(
                    alphas[row['t1_idx']], betas[row['t1_idx']],
                    alphas[row['t2_idx']], betas[row['t2_idx']],
                    gamma, rho, row['TeamHome']
                )
                score = calculate_brier_score(p_t1, p_draw, p_t2, row['GF'], row['GA'])
                total_brier += score
                predictions_made += 1
        
        current_train_end += timedelta(days=step_days)
        
    avg_brier = total_brier / max(1, predictions_made)
    return xi, avg_brier

def walk_forward_validation_parallel(df, xi_grid, num_teams, initial_days=365, step_days=60):
    best_xi = None
    best_score = float('inf')
    
    tasks = [(df, xi, num_teams, initial_days, step_days) for xi in xi_grid]
    print(f"Spinning up {len(xi_grid)} parallel processes for walk-forward evaluation...")
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(evaluate_single_xi, tasks)
        
    for xi, avg_brier in results:
        print(f"Completed xi={xi:.5f} | Average Brier Score: {avg_brier:.4f}")
        if avg_brier < best_score:
            best_score = avg_brier
            best_xi = xi
            
    return best_xi

# ==========================================
# 3. I/O AND EXECUTION
# ==========================================

def run_master_pipeline(csv_path):
    print("Loading data and mapping teams...")
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    df = df.dropna(subset=['GF', 'GA'])
    
    teams = pd.unique(df[['Team1', 'Team2']].values.ravel('K'))
    team_to_idx = {team: idx for idx, team in enumerate(teams)}
    idx_to_team = {idx: team for team, idx in team_to_idx.items()}
    num_teams = len(teams)
    
    df['t1_idx'] = df['Team1'].map(team_to_idx)
    df['t2_idx'] = df['Team2'].map(team_to_idx)

    xi_grid = [0.0077, 0.0038, 0.0019, 0.0010, 0.0006] 
    
    best_xi = walk_forward_validation_parallel(df, xi_grid, num_teams, initial_days=365, step_days=60)
    
    print(f"\nOptimization Complete. Best performing decay parameter (xi): {best_xi}")
    print("\nRunning final parameter calculation on entire dataset...")
    final_params = solve_parameters(df, best_xi, num_teams)
    
    alphas = final_params[:num_teams]
    betas = final_params[num_teams:2 * num_teams]
    gamma = final_params[-2]
    rho = final_params[-1]

    output_data = {
        "global_params": {
            "gamma": float(np.round(gamma, 4)),
            "rho": float(np.round(rho, 4)),
            "xi": float(best_xi)
        },
        "teams": []
    }

    for idx in range(num_teams):
        output_data["teams"].append({
            "name": str(idx_to_team[idx]),
            "alpha": float(np.round(alphas[idx], 4)),
            "beta": float(np.round(betas[idx], 4))
        })

    base_name = os.path.splitext(csv_path)[0]
    json_path = f"{base_name}.json"

    with open(json_path, 'w', encoding='utf-8') as json_file:
        json.dump(output_data, json_file, indent=4)
        
    print(f"Success! Model parameters exported to {json_path}")

if __name__ == "__main__":
    import sys
    target_csv = sys.argv[1] if len(sys.argv) > 1 else "data/proccessed/epl_ready_data.csv"
    start = datetime.datetime.now()
    run_master_pipeline(target_csv)
    end = datetime.datetime.now()
    print(f"Time elapsed: {(end - start).seconds}s")
