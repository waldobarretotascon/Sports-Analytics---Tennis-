# Sports-Analytics---Tennis-

# 🎾 Predicting ATP Grand Slam Winners

This project builds a smarter tennis forecasting model to predict ATP Grand Slam winners using historical data, engineered features, Elo ratings, and Monte Carlo simulations. Unlike traditional models that rely solely on ATP rankings, this system incorporates momentum, surface-specific performance, and randomness in the tournament draw — simulating how tennis actually plays out.

---

## 🧠 Problem Statement

Most tennis prediction models:
- Rely too heavily on **ATP rankings or seedings**
- Assume rankings reflect current form (which they don’t)
- Predict binary outcomes with no sense of **probability**
- Don’t simulate full tournaments — missing effects like “draw luck”

---

## 💡 Our Solution

We propose a data-driven approach with:
- Surface-aware modeling
- Probabilistic match outcomes
- Momentum and form tracking
- Full tournament simulation via Monte Carlo
- Smarter feature engineering and machine learning

---

## 📦 Project Structure

Sports-Analytics---Tennis-/
├── monte_carlo_tennis.py # Core logic: modeling, Elo, and simulation
├── streamlit_app.py # Streamlit dashboard for simulation
├── requirements.txt # Python dependencies
├── images/ # Visuals and plots
└── README.md # This file


---

## 📊 Dataset

- **Source**: Jeff Sackmann’s ATP match dataset (1968–2024)
- **Includes**: Tournament details, player stats, serve metrics, outcomes
- **Span**: Over 50 years of match data, used to engineer performance-based features

---

## 🧩 Key Features Engineered

- `elo_difference` (dynamic, surface-aware)
- `head_to_head` history
- `recent_form` (last 10 matches)
- `surface_win_rate`
- `grand_slam_win_rate`
- `age_difference`
- `seeding_difference`

Each match is added from both perspectives (winner and loser) to balance the dataset.

---

## 🧮 Model & Simulation

- **Machine Learning Model**: Random Forest Classifier
- **Training**: On historical features (2010–2024)
- **Elo Ratings**: Adjusted by surface, recency, and match importance
- **Monte Carlo Simulation**: 
    - Simulates entire tournament 200 times
    - Uses weighted average of Elo (70%) and model prediction (30%)
    - Recreates full draw structure with seeded and random players
    - Outputs tournament win probabilities for each player

---

## 🚀 How to Run

1. **Clone the repository**

```bash
git clone https://github.com/waldobarretotascon/Sports-Analytics---Tennis-.git
cd Sports-Analytics---Tennis-

2. **Install dependencies**

```bash
pip install -r requirements.txt

3. **Run the Streamlit APP**

streamlit run streamlit_app.py


