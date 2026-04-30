#!/usr/bin/env python
"""Quick test to verify the updated pipeline works."""
from src.ml_pipeline import train_all
from src.data_loader import load_dataset

print("Loading dataset...")
df = load_dataset("final_fixed_sustainability_dataset.csv", max_rows=500)

print("Training Gradient Boosting models...")
bundle = train_all(df)

print("\n✓ Models trained successfully!\n")
print("Performance Metrics:")
print(f"  Energy     | R²: {bundle.metrics['energy']['r2']:.4f} | MAE: {bundle.metrics['energy']['mae']:.2f}")
print(f"  Water      | R²: {bundle.metrics['water']['r2']:.4f} | MAE: {bundle.metrics['water']['mae']:.2f}")
print(f"  Sustainabi | R²: {bundle.metrics['sustainability']['r2']:.4f} | MAE: {bundle.metrics['sustainability']['mae']:.4f}")

print("\nModel types:")
print(f"  gb_energy:   {type(bundle.gb_energy).__name__}")
print(f"  gb_water:    {type(bundle.gb_water).__name__}")
print(f"  gb_sustain:  {type(bundle.gb_sustain).__name__}")
print(f"  lr_carbon:   {type(bundle.lr_carbon).__name__}")

print("\n✓ All components working correctly!")
