# Sports-Analytics---Tennis-

## 🎾 **Forecasting ATP Grand Slam Winners using Elo Ratings & Monte Carlo Simulation**
This project builds a smarter tennis forecasting model to predict ATP Grand Slam winners using historical data, engineered features, Elo ratings, and Monte Carlo simulations. Unlike traditional models that rely solely on ATP rankings, this system incorporates momentum, surface-specific performance, and randomness in the tournament draw — simulating how tennis actually plays out.

---

## 📚 Table of Contents

- 🎯 Problem Statement  
- 💡 Our Solution  
- 📁 Project Structure  
- 📂 Dataset  
- 🧠 Key Features Engineered  
- 🧪 Model & Simulation  
- ⚙️ Requirements  
- 🚀 How to Run the Project  
- 🧾 Example Usage  
- 👥 Team  
- 🤝 Contributing  
- 📬 Contact


## 🧠 Problem Statement

Most tennis prediction models:
- Rely too heavily on **ATP rankings or seedings**
- Assume rankings reflect current form (which they don’t)
- Predict binary outcomes with no sense of **probability**
- Don’t simulate full tournaments — missing effects like “draw luck”

---

## 💡 Our Solution

We propose a data-driven approach with:
- Surface-aware modeling**
- Probabilistic match outcomes
- Momentum and form tracking
- Full tournament simulation via Monte Carlo
- Smarter feature engineering and machine learning

---

## 📦 Project Structure

```text
Sports-Analytics---Tennis/
├── monte_carlo_tennis.py   # Core logic: modeling, Elo rating, and Monte Carlo simulation
├── streamlit_app.py        # Streamlit dashboard to visualize simulation results
├── requirements.txt        # Python dependencies for running the project
├── images/                 # Folder for visuals and plots
└── README.md               # Project documentation (this file)
```



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
### ⚙️ Requirements

- Python 3.8 or higher
- Streamlit
- pandas, numpy, scikit-learn, etc. (see `requirements.txt`)


## 🚀 How to Run the Project

Follow these steps to get the project up and running on your local machine.  
Run all the commands using the **Terminal** (go to the top menu and click: `Terminal → New Terminal`).

---

### 🔹 Step 1: Clone the Repository

Clone the repository and navigate into the project folder:

```bash
git clone https://github.com/waldobarretotascon/Sports-Analytics---Tennis-.git
cd Sports-Analytics---Tennis-
```

### 🔹 Step 2: Install the Dependencies

Install the required Python packages:

#### Install directly (Python 3.8+ required)

```bash
pip install -r requirements.txt
```
### 🔹 Step 3: Launch the Streamlit Dashboard

Run the following command to launch the Streamlit app locally:

```bash
streamlit run streamlit_app.py
```
### 🧪 Example Usage

Run a quick simulation through the terminal:
```bash
python monte_carlo_tennis.py
```

## 👥 Team

- Alejandro Osto  
- Waldo Barreto Tascon  
- Marta Pérez  
- Ignacio Salceda

### 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.

### 📬 Contact

For questions or collaborations, reach out via GitHub or email.

