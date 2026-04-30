# Smart Building Energy & ESG Predictor

An end-to-end Machine Learning web application that analyzes historical building sensor data to predict energy consumption, water usage, carbon emissions, and sustainability score.  
The dashboard also performs cost estimation, scenario testing, and actionable waste detection to support better ESG and facility management decisions.

---

## Project Overview

Traditional Building Management Systems often rely on fixed formulas and simple thresholds.  
This project takes a smarter approach by using machine learning to learn the unique behavior of a building from historical sensor data.

It predicts:

- Total Energy Consumption (kWh)
- Total Water Usage (Liters)
- Carbon Emissions
- Sustainability Score (0–100)

It also converts predictions into financial impact and helps users test “what-if” scenarios using interactive inputs such as occupancy, temperature, lighting, and HVAC load.

---

## Features

- Interactive Streamlit dashboard
- Machine learning-based prediction engine
- Gradient Boosting Regressor for structured tabular data
- RandomizedSearchCV with cross-validation for tuning
- Feature importance analysis for explainability
- Cost estimation using utility rates
- Waste and anomaly detection
- ESG-oriented sustainability insights
- Interactive Plotly charts
- Clean modular code structure

---

## Tech Stack

- **Frontend:** Streamlit
- **Visualization:** Plotly, Seaborn
- **Machine Learning:** Scikit-Learn
- **Data Processing:** Pandas, NumPy
- **Modeling:** GradientBoostingRegressor, Linear Regression
- **Optimization:** RandomizedSearchCV, Cross-Validation

---

## Project Architecture

### 1. Data Ingestion & Engineering (`data_loader.py`)
This module loads the historical sensor dataset, cleans it, and prepares it for model training.

Key responsibilities:
- Load raw CSV data
- Remove noisy or synthetic early-stage records
- Filter data before March 2020
- Prepare a clean dataset for training and evaluation

> Garbage In = Garbage Out

---

### 2. Machine Learning Engine (`ml_pipeline.py`)
This module trains the prediction models and generates performance metrics.

Key responsibilities:
- Train a `GradientBoostingRegressor`
- Tune hyperparameters using `RandomizedSearchCV`
- Apply 3-fold cross-validation
- Predict energy, water, sustainability score, and carbon emissions
- Use a Linear Regression sub-model for emissions

Selected model configuration:
- `n_estimators=200`
- `learning_rate=0.05`
- `max_depth=3`

---

### 3. Financial & Alert Logic (`insights.py`)
This module converts predictions into practical insights.

Key responsibilities:
- Calculate estimated utility cost
- Detect wasteful or illogical input combinations
- Highlight potential daily financial loss
- Suggest corrective actions such as motion sensors or load optimization

Example:
If lighting usage is very high but occupancy is very low, the system flags it as wasted consumption.

---

### 4. Dashboard (`app.py`)
This is the Streamlit interface where users interact with the model.

Features:
- User input sliders
- Real-time prediction output
- Cost breakdown
- Sustainability insights
- Prediction charts
- Feature importance visualization

---

## How It Works

1. User enters building conditions using the dashboard sliders.
2. The system formats the inputs into a valid DataFrame.
3. The trained ML model predicts energy, water, emissions, and sustainability score.
4. The app calculates financial cost using utility rates.
5. The system checks for waste patterns and generates actionable insights.
6. Results are displayed visually in the dashboard.

---

## Model Performance

The model was tuned using randomized hyperparameter search and cross-validation to reduce overfitting and improve generalization.

Reported performance:
- Energy prediction accuracy: over 84%
- Water prediction accuracy: over 98%

These results show that the model learned meaningful building patterns from the historical dataset.

---

## Why Gradient Boosting?

Gradient Boosting works well for structured tabular data because it:
- Handles non-linear relationships effectively
- Performs strongly on small-to-medium datasets
- Provides feature importance for explainability
- Often outperforms more complex deep learning approaches on CSV-style business data

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
