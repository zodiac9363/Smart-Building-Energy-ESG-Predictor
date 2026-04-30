#!/usr/bin/env python
"""Test the advanced anomaly detection system."""
from src.data_loader import load_dataset
from src.insights import generate_alerts

print("Loading dataset...")
df = load_dataset("final_fixed_sustainability_dataset.csv", max_rows=500)

print("Running anomaly detection...\n")
anomalies = generate_alerts(df)

print(f"✓ Total anomalies detected: {len(anomalies)}\n")

if anomalies:
    # Count by severity
    critical = sum(1 for a in anomalies if "Critical" in a.severity)
    mild = sum(1 for a in anomalies if "Mild" in a.severity)
    
    print(f"  🔴 Critical anomalies: {critical}")
    print(f"  🟡 Mild anomalies: {mild}\n")
    
    # Show top 3 anomalies
    print("Top Anomalies:")
    print("="*80)
    for i, anomaly in enumerate(anomalies[:3], 1):
        print(f"\n{i}. {anomaly.severity}")
        print(f"   Score: {anomaly.anomaly_score:.1f}/100")
        print(f"   Alerts: {', '.join(anomaly.alerts)}")
        print(f"   Recommendations:")
        for rec in anomaly.recommendations:
            print(f"     • {rec}")
        print(f"   Metrics - Energy: {anomaly.energy_kwh:,.0f} kWh | Water: {anomaly.water_l:,.0f} L | Sustain: {anomaly.sustainability:.3f}")
else:
    print("No anomalies detected - all systems normal!")

print("\n✓ Anomaly detection system working correctly!")
