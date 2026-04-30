from src.data_loader import load_dataset, DEFAULT_CSV
from src.ml_pipeline import train_all

# Load data
df = load_dataset(str(DEFAULT_CSV), max_rows=1000)

# Train models
bundle = train_all(df)

# Print MAE scores
print("MAE Scores:")
for key, metric in bundle.metrics.items():
    print(f"{key.title()}: {metric['mae']:.2f}")

# Also print R² for context
print("\nR² Scores:")
for key, metric in bundle.metrics.items():
    print(f"{key.title()}: {metric['r2']:.3f}")