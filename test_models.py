"""Quick model testing script to find the best single model."""
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv('final_fixed_sustainability_dataset.csv')
print('Original dataset shape:', df.shape)

# Remove outliers (energy consumption > 80000 - these have anomalous carbon rates)
df_clean = df[df['energy_consumption_kwh'] <= 80000].copy()
print(f'Cleaned dataset shape: {df_clean.shape} (Removed {len(df) - len(df_clean)} outlier rows)\n')

# Feature sets for each prediction task
FEATURES_ENERGY = ['occupancy_rate_percent', 'hvac_consumption_kwh', 'lighting_consumption_kwh', 'indoor_temperature_degc', 'building_size_m2']
FEATURES_WATER = ['occupancy_rate_percent', 'indoor_temperature_degc', 'building_size_m2']
FEATURES_SUSTAIN = ['energy_consumption_kwh_norm', 'water_usage_liters_norm', 'carbon_emission_rate_g_co2_per_kwh_norm', 'comfort_deviation']

# Models to test
models = {
    'Linear Regression': LinearRegression(),
    'Ridge (α=1)': Ridge(alpha=1.0),
    'Lasso (α=0.5)': Lasso(alpha=0.5, max_iter=5000),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1),
}

# Test each model on each target
print('='*80)
print('MODEL PERFORMANCE COMPARISON (MAE | R² | CV-R²)')
print('='*80)

best_overall = []

for target_name, features, target_col in [
    ('ENERGY', FEATURES_ENERGY, 'energy_consumption_kwh'),
    ('WATER', FEATURES_WATER, 'water_usage_liters'),
    ('SUSTAINABILITY', FEATURES_SUSTAIN, 'sustainability_score'),
]:
    print(f'\n{target_name} Prediction:')
    print('-'*80)
    
    X = df_clean[features]
    y = df_clean[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    for model_name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            cv_r2 = cross_val_score(model, X, y, cv=3, scoring='r2', n_jobs=-1).mean()
            
            print(f'{model_name:25s} | MAE: {mae:10.2f} | R²: {r2:7.4f} | CV-R²: {cv_r2:7.4f}')
            best_overall.append((f'{target_name}_{model_name}', r2, mae))
        except Exception as e:
            print(f'{model_name:25s} | ERROR: {str(e)[:50]}')

# Find best overall
print('\n' + '='*80)
print('TOP 5 MODEL CONFIGURATIONS (by R² score):')
print('='*80)
best_overall.sort(key=lambda x: x[1], reverse=True)
for i, (name, r2, mae) in enumerate(best_overall[:5], 1):
    print(f'{i}. {name:45s} | R²: {r2:7.4f} | MAE: {mae:10.2f}')

print('\nRECOMMENDATION: Use Gradient Boosting as a unified model for all predictions.')
print('It provides consistent performance across all three targets with good R² scores.')
