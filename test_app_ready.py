#!/usr/bin/env python
"""Quick test to verify app imports and basic functionality."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from src.data_loader import load_dataset, DEFAULT_CSV
    from src.insights import generate_alerts, Anomaly
    from src.ml_pipeline import train_all, predict_from_sliders
    
    print("✓ All imports successful")
    
    # Load data
    df = load_dataset(str(DEFAULT_CSV), max_rows=300)
    print(f"✓ Data loaded: {len(df)} rows")
    
    # Train models
    bundle = train_all(df)
    print(f"✓ Models trained")
    
    # Generate anomalies
    anomalies = generate_alerts(df)
    print(f"✓ Anomaly detection: {len(anomalies)} anomalies found")
    
    # Test predictions
    pred = predict_from_sliders(bundle, 55, 22.5, 16, 11, 1100)
    print(f"✓ Predictions working")
    print(f"  - Energy: {pred['energy_kwh']:,.0f} kWh")
    print(f"  - Water: {pred['water_l']:,.0f} L")
    print(f"  - Sustainability: {pred['sustainability']:.3f}")
    
    print("\n✅ App is ready for streaming!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
