import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV

from src.data_loader import load_dataset
from src.ml_pipeline import FEATURES_ENERGY

def main():
    print("Loading data...")
    df = load_dataset()
    X = df[FEATURES_ENERGY]
    y = df["energy_consumption_kwh"]

    print("Setting up RandomizedSearchCV...")
    param_grid = {
        "n_estimators": [50, 100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 4, 5, 6, 8],
        "min_samples_leaf": [2, 4, 10],
        "subsample": [0.8, 0.9, 1.0]
    }

    gb = GradientBoostingRegressor(random_state=42)
    # 20 iterations, 3-fold CV -> 60 fits
    search = RandomizedSearchCV(
        estimator=gb,
        param_distributions=param_grid,
        n_iter=20,
        scoring="r2",
        cv=3,
        verbose=2,
        random_state=42,
        n_jobs=-1
    )

    print("Running optimization...")
    search.fit(X, y)

    print("\n=== OPTIMIZATION COMPLETE ===")
    print(f"Best R2 Score: {search.best_score_:.4f}")
    print("Best Parameters:")
    for k, v in search.best_params_.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
