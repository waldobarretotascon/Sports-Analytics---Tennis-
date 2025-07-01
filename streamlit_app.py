import streamlit as st
import pandas as pd
import numpy as np
from monte_carlo_tennis import load_data, calculate_elo, create_features, train_model, monte_carlo
import streamlit.components.v1 as components

st.set_page_config(page_title="Tennis Grand Slam Winner Predictor", layout="centered")
st.title('🎾 Tennis Grand Slam Winner Predictor')

# Tournament options
grand_slams = [
    {'name': 'Australian Open 2025', 'surface': 'Hard'},
    {'name': 'Roland Garros 2025', 'surface': 'Clay'},
    {'name': 'Wimbledon 2025', 'surface': 'Grass'},
    {'name': 'US Open 2025', 'surface': 'Hard'}
]
slam_names = [slam['name'] for slam in grand_slams]

# Sidebar for tournament selection
st.sidebar.header("Simulation Settings")
selected_slam = st.sidebar.selectbox('Select Grand Slam', slam_names)
slam = next(s for s in grand_slams if s['name'] == selected_slam)
surface = slam['surface']

n_simulations = st.sidebar.slider('Number of Simulations', 50, 500, 200, step=50)

# Load and cache data/model
@st.cache_data(show_spinner=True)
def get_model_data():
    df = load_data(start_year=2020)
    player_elo, elo_history = calculate_elo(df)
    features_df = create_features(df, player_elo)
    required_features = [
        'seed_diff', 'elo_diff', 'win_rate_diff', 'surface_win_rate_diff',
        'form_diff', 'h2h_diff', 'age_diff', 'gs_deep_runs_diff',
        'surface_Hard', 'surface_Clay', 'surface_Grass'
    ]
    X = features_df[required_features]
    y = features_df['winner']
    model = train_model(X, y)
    return df, model, player_elo

df, model, player_elo = get_model_data()

# Main UI
st.markdown(f"**Tournament:** {selected_slam}")
st.markdown(f"**Surface:** {surface}")

# Show actual bracket image for Australian Open 2025
if selected_slam == "Australian Open 2025":
    st.subheader("Actual Australian Open 2025 Bracket")
    st.image("images/australian_open_2025_bracket.png", caption="Australian Open 2025 Bracket", use_container_width=True)

# Show actual bracket image for Roland Garros 2025
if selected_slam == "Roland Garros 2025":
    st.subheader("Actual Roland Garros 2025 Bracket")
    st.image("images/roland_garros_bracket.png", caption="Roland Garros 2025 Bracket", use_container_width=True)

# Show actual bracket image for Wimbledon 2025
if selected_slam == "Wimbledon 2025":
    st.subheader("Actual Wimbledon 2025 Bracket")
    st.image("images/wimblendon_odds.png", caption="Wimbledon 2025 Bracket", use_container_width=True)

# Show actual bracket image for US Open 2025
if selected_slam == "US Open 2025":
    st.subheader("Actual US Open 2025 Bracket")
    st.image("images/US_open_odds.png", caption="US Open 2025 Bracket", use_container_width=True)

if st.button('🎲 Simulate Tournament'):
    with st.spinner('Running Monte Carlo simulation...'):
        results = monte_carlo(df, model, player_elo, surface, selected_slam, n_simulations=n_simulations)
        preds = results['predictions']
        # Prepare DataFrame for display
        df_preds = pd.DataFrame([
            {'Player': player, 'Win Probability': prob, 'Odds': 1/prob if prob > 0 else np.nan}
            for player, prob in preds.items()
        ])
        st.subheader('Top 10 Predicted Winners')
        st.dataframe(df_preds.style.format({'Win Probability': '{:.1%}', 'Odds': '{:.2f}'}), use_container_width=True)
        st.subheader('Win Probability Bar Chart')
        st.bar_chart(df_preds.set_index('Player')['Win Probability'])
else:
    st.info('Select a tournament and click "Simulate Tournament" to see predictions.') 