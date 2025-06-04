import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from collections import defaultdict
import random
import datetime

# 1. Load and Preprocess Data
def load_data(start_year=2020):
    # List to store dataframes
    dfs = []
    current_year = datetime.datetime.now().year
    
    print(f"Attempting to load data from {start_year} to {current_year}")
    
    # Load data from start_year to current year
    for year in range(start_year, current_year + 1):
        url = f'https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{year}.csv'
        try:
            print(f"Loading data for {year}...")
            df = pd.read_csv(url)
            print(f"Successfully loaded {len(df)} matches from {year}")
            dfs.append(df)
        except Exception as e:
            print(f"Could not load data for {year}: {e}")
    
    if not dfs:
        raise ValueError("No data could be loaded. Please check your internet connection or data source.")
    
    # Combine all dataframes
    print("Combining data from all years...")
    df = pd.concat(dfs, ignore_index=True)
    
    # Filter Grand Slams and add tournament names
    df = df[df['tourney_level'] == 'G']
    print(f"Found {len(df)} Grand Slam matches")
    
    # Convert date and create new features
    df['tourney_date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d')
    df['year'] = df['tourney_date'].dt.year
    
    # Add age at tournament
    df['winner_age_at_tournament'] = df['winner_age']
    df['loser_age_at_tournament'] = df['loser_age']
    
    # Print some statistics
    print("\nData Summary:")
    print(f"Years covered: {df['year'].min()} to {df['year'].max()}")
    print(f"Number of unique players: {len(set(df['winner_name'].unique()) | set(df['loser_name'].unique()))}")
    print(f"Surfaces: {df['surface'].unique()}")
    
    return df.sort_values('tourney_date').reset_index(drop=True)

# 2. Elo Rating System
def calculate_elo(df, k=32, default_elo=1500):
    player_elo = defaultdict(lambda: defaultdict(lambda: default_elo))
    elo_history = defaultdict(list)
    
    # Add recency weighting to k-factor
    current_year = datetime.datetime.now().year
    
    # Tournament stage importance factors
    stage_importance = {
        'F': 2.0,    # Final
        'SF': 1.7,   # Semi-final
        'QF': 1.5,   # Quarter-final
        'R16': 1.3,  # Round of 16
        'R32': 1.2,  # Round of 32
        'R64': 1.1,  # Round of 64
        'R128': 1.0  # Round of 128
    }
    
    # Tournament level importance
    tourney_importance = {
        'G': 2.0,    # Grand Slam
        'M': 1.5,    # Masters 1000
        'A': 1.3,    # ATP 500
        'B': 1.1,    # ATP 250
        'D': 1.0     # Other tournaments
    }
    
    for idx, row in df.iterrows():
        winner = row['winner_name']
        loser = row['loser_name']
        surface = row['surface']
        match_year = row['year']
        round_name = row['round']
        tourney_level = row['tourney_level']
        
        # Calculate importance multiplier
        stage_factor = stage_importance.get(round_name, 1.0)
        tourney_factor = tourney_importance.get(tourney_level, 1.0)
        importance_multiplier = stage_factor * tourney_factor
        
        # Increase k-factor for more recent matches and younger players
        winner_age = row['winner_age'] if pd.notnull(row['winner_age']) else 25
        loser_age = row['loser_age'] if pd.notnull(row['loser_age']) else 25
        
        # Higher k-factor for:
        # 1. Recent matches
        # 2. Young players (higher potential for improvement)
        # 3. Important matches (Grand Slams, later rounds)
        recency_factor = 1 + (match_year - 2020) * 0.2
        age_factor_winner = 1.2 if winner_age < 23 else 1.0
        age_factor_loser = 1.2 if loser_age < 23 else 1.0
        
        k_winner = k * recency_factor * age_factor_winner * importance_multiplier
        k_loser = k * recency_factor * age_factor_loser * importance_multiplier
        
        # Get current Elo ratings
        winner_elo = player_elo[winner][surface]
        loser_elo = player_elo[loser][surface]
        
        # Calculate expected scores
        expected_winner = 1 / (1 + 10**((loser_elo - winner_elo)/400))
        expected_loser = 1 - expected_winner
        
        # Update Elo ratings
        player_elo[winner][surface] += k_winner * (1 - expected_winner)
        player_elo[loser][surface] += k_loser * (0 - expected_loser)
        
        # Store history for visualization
        elo_history[winner].append((row['tourney_date'], surface, player_elo[winner][surface]))
        elo_history[loser].append((row['tourney_date'], surface, player_elo[loser][surface]))
    
    return dict(player_elo), elo_history

# 3. Feature Engineering
def create_features(df, player_elo):
    features = []
    
    # Create player statistics dictionary
    player_stats = defaultdict(lambda: {
        'matches_played': 0,
        'matches_won': 0,
        'grand_slam_matches': 0,
        'grand_slam_wins': 0,
        'surface_matches': defaultdict(int),
        'surface_wins': defaultdict(int),
        'recent_matches': [],  # Store last 10 matches
        'h2h_records': defaultdict(lambda: {'wins': 0, 'losses': 0}),
        'grand_slam_performance': defaultdict(lambda: {'matches': 0, 'wins': 0}),  # Track performance by round
        'recent_grand_slam_results': []  # Store last 5 Grand Slam results with round reached
    })
    
    # First pass: collect player statistics
    for idx, row in df.iterrows():
        winner = row['winner_name']
        loser = row['loser_name']
        surface = row['surface']
        round_name = row['round']
        tourney_level = row['tourney_level']
        
        # Update general stats
        player_stats[winner]['matches_played'] += 1
        player_stats[winner]['matches_won'] += 1
        player_stats[loser]['matches_played'] += 1
        
        # Update Grand Slam stats with round information
        if tourney_level == 'G':
            player_stats[winner]['grand_slam_matches'] += 1
            player_stats[winner]['grand_slam_wins'] += 1
            player_stats[loser]['grand_slam_matches'] += 1
            
            # Track performance by round
            player_stats[winner]['grand_slam_performance'][round_name]['matches'] += 1
            player_stats[winner]['grand_slam_performance'][round_name]['wins'] += 1
            player_stats[loser]['grand_slam_performance'][round_name]['matches'] += 1
            
            # Store recent Grand Slam results
            player_stats[winner]['recent_grand_slam_results'].append((round_name, 'W'))
            player_stats[loser]['recent_grand_slam_results'].append((round_name, 'L'))
            player_stats[winner]['recent_grand_slam_results'] = player_stats[winner]['recent_grand_slam_results'][-5:]
            player_stats[loser]['recent_grand_slam_results'] = player_stats[loser]['recent_grand_slam_results'][-5:]
        
        # Update surface stats
        player_stats[winner]['surface_matches'][surface] += 1
        player_stats[winner]['surface_wins'][surface] += 1
        player_stats[loser]['surface_matches'][surface] += 1
        
        # Update head-to-head records
        player_stats[winner]['h2h_records'][loser]['wins'] += 1
        player_stats[loser]['h2h_records'][winner]['losses'] += 1
        
        # Update recent matches (keep last 10)
        player_stats[winner]['recent_matches'].append(1)
        player_stats[loser]['recent_matches'].append(0)
        player_stats[winner]['recent_matches'] = player_stats[winner]['recent_matches'][-10:]
        player_stats[loser]['recent_matches'] = player_stats[loser]['recent_matches'][-10:]
    
    # Second pass: create feature vectors
    for idx, row in df.iterrows():
        winner = row['winner_name']
        loser = row['loser_name']
        surface = row['surface']
        
        # Calculate win rates with recency weighting
        winner_recent_form = sum(player_stats[winner]['recent_matches']) / len(player_stats[winner]['recent_matches']) if player_stats[winner]['recent_matches'] else 0.5
        loser_recent_form = sum(player_stats[loser]['recent_matches']) / len(player_stats[loser]['recent_matches']) if player_stats[loser]['recent_matches'] else 0.5
        
        # Calculate Grand Slam performance metrics
        winner_gs_deep_runs = sum(1 for round_name, result in player_stats[winner]['recent_grand_slam_results'] 
                                if round_name in ['QF', 'SF', 'F'])
        loser_gs_deep_runs = sum(1 for round_name, result in player_stats[loser]['recent_grand_slam_results'] 
                               if round_name in ['QF', 'SF', 'F'])
        
        # Calculate surface-specific win rates
        winner_surface_wr = player_stats[winner]['surface_wins'][surface] / max(1, player_stats[winner]['surface_matches'][surface])
        loser_surface_wr = player_stats[loser]['surface_wins'][surface] / max(1, player_stats[loser]['surface_matches'][surface])
        
        # Head-to-head stats
        h2h_winner = player_stats[winner]['h2h_records'][loser]['wins']
        h2h_loser = player_stats[loser]['h2h_records'][winner]['wins']
        
        # Age-based potential factor (higher for young players)
        winner_age = row['winner_age'] if pd.notnull(row['winner_age']) else 25
        loser_age = row['loser_age'] if pd.notnull(row['loser_age']) else 25
        winner_potential = max(1.0, 1.3 - winner_age/30)  # Higher potential for younger players
        loser_potential = max(1.0, 1.3 - loser_age/30)
        
        # Create feature vectors for both winner and loser perspective
        base_features = {
            'seed_diff': row['winner_seed'] - row['loser_seed'] if pd.notnull(row['winner_seed']) and pd.notnull(row['loser_seed']) else 0,
            'elo_diff': (player_elo[winner][surface] - player_elo[loser][surface]) * winner_potential / loser_potential,
            'win_rate_diff': winner_recent_form - loser_recent_form,
            'surface_win_rate_diff': winner_surface_wr - loser_surface_wr,
            'form_diff': winner_recent_form - loser_recent_form,
            'h2h_diff': h2h_winner - h2h_loser,
            'age_diff': winner_age - loser_age,
            'gs_deep_runs_diff': winner_gs_deep_runs - loser_gs_deep_runs,
            'surface_Hard': 1 if surface == 'Hard' else 0,
            'surface_Clay': 1 if surface == 'Clay' else 0,
            'surface_Grass': 1 if surface == 'Grass' else 0,
            'winner': 1
        }
        
        features.append(base_features.copy())
        
        # Add reverse case
        reverse_features = base_features.copy()
        for key in ['seed_diff', 'elo_diff', 'win_rate_diff', 'surface_win_rate_diff', 'form_diff', 
                   'h2h_diff', 'age_diff', 'gs_deep_runs_diff']:
            reverse_features[key] = -reverse_features[key]
        reverse_features['winner'] = 0
        
        features.append(reverse_features)
    
    return pd.DataFrame(features)

# 4. Train Match Prediction Model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    preds = model.predict(X_test)
    print(f"Model Accuracy: {accuracy_score(y_test, preds):.2f}")
    
    return model

# 5. Tournament Simulation
def simulate_match(player1, player2, surface, model, player_elo):
    # Calculate Elo-based probability
    elo_diff = player_elo[player1][surface] - player_elo[player2][surface]
    elo_prob = 1 / (1 + 10**(-elo_diff/400))
    
    # Create feature vector with all required features
    features = pd.DataFrame([{
        'seed_diff': 0,  # We don't have seeding info for future tournaments
        'elo_diff': elo_diff,  # Use actual Elo difference
        'win_rate_diff': 0.2 * (elo_prob - 0.5),  # Derive from Elo
        'surface_win_rate_diff': 0.3 * (elo_prob - 0.5) if surface in ['Clay', 'Grass'] else 0.1 * (elo_prob - 0.5),
        'form_diff': 0.15 * (elo_prob - 0.5),  # Recent form contribution
        'h2h_diff': 0,  # Historical head-to-head captured in Elo
        'age_diff': 0,  # Age factor already applied in Elo calculation
        'gs_deep_runs_diff': 0.25 * (elo_prob - 0.5),  # Scale with Elo rating
        'surface_Hard': 1 if surface == 'Hard' else 0,
        'surface_Clay': 1 if surface == 'Clay' else 0,
        'surface_Grass': 1 if surface == 'Grass' else 0
    }])
    
    try:
        # Combine model probability with Elo probability
        model_prob = model.predict_proba(features)[0][1]
        final_prob = 0.7 * elo_prob + 0.3 * model_prob  # Give more weight to Elo
        return player1 if random.random() < final_prob else player2
    except Exception as e:
        print(f"Error in match simulation between {player1} and {player2}: {e}")
        # Use pure Elo rating as fallback
        return player1 if random.random() < elo_prob else player2

def run_tournament(players, surface, model, player_elo):
    current_round = players.copy()
    
    while len(current_round) > 1:
        next_round = []
        for i in range(0, len(current_round), 2):
            p1 = current_round[i]
            p2 = current_round[i+1] if i+1 < len(current_round) else p1  # Bye case
            
            if p1 == p2:  # Handle byes
                next_round.append(p1)
            else:
                winner = simulate_match(p1, p2, surface, model, player_elo)
                next_round.append(winner)
        current_round = next_round
    
    return current_round[0]

# 6. Monte Carlo Simulation
def monte_carlo(df, model, player_elo, surface, tournament_name, n_simulations=200):
    print(f"\nPreparing {tournament_name} simulation...")
    
    # Get players from recent years with more weight on recent performance
    current_year = datetime.datetime.now().year
    df['recency_weight'] = (df['year'] - (current_year - 5)) / 5
    df['recency_weight'] = df['recency_weight'].clip(0, 1)
    
    # List of known retired players
    retired_players = {
        'Rafael Nadal',
        'Roger Federer',
        'Juan Martin del Potro',
        'Jo-Wilfried Tsonga',
        'John Isner',
        'Kevin Anderson',
        'Philipp Kohlschreiber'
        # Add more retired players as needed
    }
    
    # Get unique players with their latest rankings and performance
    recent_players = set()
    
    # Only consider matches from 2023 onwards for active player detection
    very_recent_df = df[df['year'] >= current_year - 1]
    recent_players.update(very_recent_df['winner_name'])
    recent_players.update(very_recent_df['loser_name'])
    
    # Filter out retired players and ensure recent activity
    active_players = {
        player for player in recent_players
        if (player not in retired_players and
            # Must have played at least 2 matches in the last year
            len(df[(df['year'] >= current_year - 1) &
                  ((df['winner_name'] == player) | (df['loser_name'] == player))]) >= 2)
    }
    
    print(f"Found {len(active_players)} active players")
    
    # Calculate player metrics with surface-specific Elo as primary factor
    player_metrics = {}
    for player in active_players:
        recent_matches = df[
            ((df['winner_name'] == player) | (df['loser_name'] == player)) &
            (df['year'] >= current_year - 1)
        ]
        
        if len(recent_matches) > 0:
            # Calculate win rate
            wins = len(recent_matches[recent_matches['winner_name'] == player])
            win_rate = wins / len(recent_matches)
            
            # Calculate surface-specific performance
            surface_matches = recent_matches[recent_matches['surface'] == surface]
            surface_wins = len(surface_matches[surface_matches['winner_name'] == player])
            surface_win_rate = surface_wins / max(1, len(surface_matches))
            
            # Get player's age and calculate potential
            player_age = None
            if len(recent_matches[recent_matches['winner_name'] == player]) > 0:
                player_age = recent_matches[recent_matches['winner_name'] == player].iloc[0]['winner_age']
            elif len(recent_matches[recent_matches['loser_name'] == player]) > 0:
                player_age = recent_matches[recent_matches['loser_name'] == player].iloc[0]['loser_age']
            
            age_factor = 1.0
            if player_age and player_age < 23:
                age_factor = 1.2  # Boost young players
            
            # Calculate composite score with more weight on Elo
            player_metrics[player] = {
                'elo': player_elo[player][surface],
                'win_rate': win_rate,
                'surface_win_rate': surface_win_rate,
                'recent_matches': len(recent_matches),
                'age_factor': age_factor,
                'composite_score': (
                    player_elo[player][surface] * 0.7 +  # More weight on Elo
                    win_rate * 1000 * 0.15 +
                    surface_win_rate * 1000 * 0.15
                ) * age_factor
            }
    
    # Sort players by composite score
    ranked_players = sorted(
        [(p, m['composite_score']) for p, m in player_metrics.items()],
        key=lambda x: x[1],
        reverse=True
    )
    
    # Print some active player stats
    print("\nTop 10 active players by rating:")
    for player, score in ranked_players[:10]:
        print(f"{player:<30} (Rating: {player_metrics[player]['elo']:.0f}, "
              f"Recent matches: {player_metrics[player]['recent_matches']})")
    
    # Take top 128 players for Grand Slam simulation
    tournament_players = [p[0] for p in ranked_players[:128]]
    
    results = defaultdict(int)
    print(f"\nRunning {n_simulations} simulations...")
    update_interval = max(1, n_simulations // 10)
    
    for sim in range(n_simulations):
        if sim % update_interval == 0:
            print(f"Progress: {sim}/{n_simulations} simulations completed ({sim/n_simulations*100:.0f}%)")
        
        # Maintain some seeding structure for realism
        top_32 = tournament_players[:32]
        rest = tournament_players[32:]
        random.shuffle(rest)
        
        # Distribute seeds throughout the draw
        seeded_players = []
        sections = 32  # Number of sections in the draw
        for i in range(sections):
            if i < len(top_32):
                seeded_players.append(top_32[i])  # Add a seeded player
            if i < len(rest):
                seeded_players.append(rest[i])    # Add an unseeded player
        
        winner = run_tournament(seeded_players, surface, model, player_elo)
        results[winner] += 1
    
    print(f"Simulation completed: {n_simulations}/{n_simulations} (100%)")
    
    # Calculate probabilities
    total = sum(results.values())
    probabilities = {k: v/total for k, v in sorted(results.items(), key=lambda x: x[1], reverse=True)}
    
    return {
        'tournament': tournament_name,
        'surface': surface,
        'year': 2025,
        'predictions': {k: v for k, v in list(probabilities.items())[:10]}  # Top 10 predictions
    }

def print_results(results, title):
    print("\n" + "="*50)
    print(title)
    print("="*50)
    for player, prob in results.items():
        print(f"{player:<30} {prob:.1%}")
    print("="*50 + "\n")

# Main Execution
if __name__ == "__main__":
    print("Loading historical tennis data...")
    df = load_data(start_year=2020)  # Load last 4 years of data
    
    print("Calculating Elo ratings...")
    player_elo, elo_history = calculate_elo(df)
    
    print("Creating features for match prediction...")
    features_df = create_features(df, player_elo)
    
    # Ensure all required features are present
    required_features = [
        'seed_diff',
        'elo_diff',
        'win_rate_diff',
        'surface_win_rate_diff',
        'form_diff',
        'h2h_diff',
        'age_diff',
        'gs_deep_runs_diff',
        'surface_Hard',
        'surface_Clay',
        'surface_Grass'
    ]
    
    # Drop the target variable and any other columns not in required_features
    X = features_df[required_features]
    y = features_df['winner']
    
    print("Training prediction model...")
    model = train_model(X, y)
    
    # Define 2025 Grand Slams
    grand_slams_2025 = [
        {'name': 'Australian Open 2025', 'surface': 'Hard'},
        {'name': 'Roland Garros 2025', 'surface': 'Clay'},
        {'name': 'Wimbledon 2025', 'surface': 'Grass'},
        {'name': 'US Open 2025', 'surface': 'Hard'}
    ]
    
    # Run Monte Carlo simulations for each 2025 Grand Slam
    all_predictions = []
    
    for slam in grand_slams_2025:
        print(f"\nSimulating {slam['name']}...")
        results = monte_carlo(df, model, player_elo, slam['surface'], slam['name'], n_simulations=200)
        all_predictions.append(results)
        print_results(results['predictions'], f"{slam['name']} Winner Prediction")
    
    # Print summary of all tournaments
    print("\nSUMMARY OF 2025 GRAND SLAM PREDICTIONS")
    print("="*50)
    for prediction in all_predictions:
        print(f"\n{prediction['tournament']}:")
        print(f"Surface: {prediction['surface']}")
        print("\nTop 5 Predicted Winners:")
        for i, (player, prob) in enumerate(list(prediction['predictions'].items())[:5], 1):
            print(f"{i}. {player:<30} {prob:.1%}")
        print("-"*50)

