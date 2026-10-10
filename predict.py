"""
Dixon-Coles Match Predictor
Reads optimal parameters from JSON and generates match probabilities.
"""

import json
import math

def load_parameters(json_path):
    """Loads the trained parameters from the JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_team_params(data, team_name):
    """Extracts alpha (attack) and beta (defense) for a specific team."""
    for team in data['teams']:
        if team['name'] == team_name:
            return team['alpha'], team['beta']
    raise ValueError(f"Team '{team_name}' not found in the dataset. Check spelling.")

def poisson_probability(lmbda, x):
    """Calculates independent Poisson probability for a single score."""
    return (math.pow(lmbda, x) * math.exp(-lmbda)) / math.factorial(x)

def predict_match(json_path, team1, team2, is_neutral=True, max_goals=10):
    """
    Builds the probability matrix and sums outcomes.
    is_neutral=True ignores the global gamma (Home Advantage) parameter.
    """
    data = load_parameters(json_path)
    
    # Extract global parameters
    rho = data['global_params']['rho']
    gamma = data['global_params']['gamma']
    
    # Extract team parameters
    alpha_1, beta_1 = get_team_params(data, team1)
    alpha_2, beta_2 = get_team_params(data, team2)
    
    # Calculate Expected Goals (Neutral venue forces gamma_1 to 1.0)
    gamma_1 = 1.0 if is_neutral else gamma
    gamma_2 = 1.0
    
    lambda_val = alpha_1 * beta_2 * gamma_1
    mu_val = alpha_2 * beta_1 * gamma_2
    
    # Initialize outcome buckets
    prob_t1_win = 0.0
    prob_draw = 0.0
    prob_t2_win = 0.0
    score_matrix = {}
    
    # Build the bivariate matrix
    for x in range(max_goals):
        for y in range(max_goals):
            p = poisson_probability(lambda_val, x) * poisson_probability(mu_val, y)
            
            # Apply the Dixon-Coles low-score correction (fixed formulation)
            if x == 0 and y == 0: p *= max(0, 1 - lambda_val * mu_val * rho)
            elif x == 1 and y == 0: p *= max(0, 1 + mu_val * rho)
            elif x == 0 and y == 1: p *= max(0, 1 + lambda_val * rho)
            elif x == 1 and y == 1: p *= max(0, 1 - rho)
            
            score_matrix[f"{x}-{y}"] = p
            
    # Normalize score matrix so total probability sums strictly to 1.0
    total_prob = sum(score_matrix.values())
    for k in score_matrix:
        score_matrix[k] /= total_prob
        x_val, y_val = map(int, k.split('-'))
        if x_val > y_val: prob_t1_win += score_matrix[k]
        elif x_val == y_val: prob_draw += score_matrix[k]
        else: prob_t2_win += score_matrix[k]
            
    return prob_t1_win, prob_draw, prob_t2_win, score_matrix, lambda_val, mu_val

# ==========================================
# EXECUTION BLOCK
# ==========================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Dixon-Coles Match Predictor")
    parser.add_argument('team1', nargs='?', default="Arsenal", help="Home team name")
    parser.add_argument('team2', nargs='?', default="Chelsea", help="Away team name")
    parser.add_argument('--league', choices=['epl', 'laliga'], default='epl', help="League JSON file to use")
    parser.add_argument('--neutral', action='store_true', help="Whether the match is on neutral ground")
    args = parser.parse_args()

    json_map = {
        'epl': "data/proccessed/epl_ready_data.json",
        'laliga': "data/proccessed/laliga_ready_data.json",
    }
    json_file = json_map[args.league]
    
    t1 = args.team1
    t2 = args.team2
    
    try:
        p_win, p_draw, p_loss, matrix, xg1, xg2 = predict_match(
            json_file, t1, t2, is_neutral=args.neutral
        )
        
        venue_str = "Neutral Venue" if args.neutral else f"Home: {t1} | Away: {t2}"
        print(f"\n--- MATCH PREDICTION: {t1} vs {t2} ({venue_str}) ---")
        print(f"Database: {json_file}")
        print(f"Expected Goals (xG): {t1} {xg1:.2f} - {xg2:.2f} {t2}\n")
        
        print("MATCH PROBABILITIES:")
        print(f"{t1} Win: {p_win * 100:.1f}%")
        print(f"Draw:      {p_draw * 100:.1f}%")
        print(f"{t2} Win: {p_loss * 100:.1f}%\n")
        
        print("TOP 3 MOST LIKELY EXACT SCORELINES:")
        sorted_scores = sorted(matrix.items(), key=lambda item: item[1], reverse=True)
        for i in range(3):
            score, prob = sorted_scores[i]
            print(f"  {score}: {prob * 100:.1f}%")
        print()
            
    except Exception as e:
        print(f"Error: {e}\n(Double-check your file path and team spelling!)")