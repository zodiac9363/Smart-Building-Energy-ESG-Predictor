"""
AI-Powered Smart Building Sustainability & Resource Optimization System
======================================================================
Run from project directory:
    streamlit run app.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import plotly.graph_objects as go
import pandas as pd
import seaborn as sns
import streamlit as st

# Project root on sys.path (same folder as this file)
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data_loader import (  # noqa: E402
    DEFAULT_CSV,
    kpi_series,
    load_dataset,
    summary_statistics,
)
from src.insights import generate_decision_alerts  # noqa: E402
from src.ml_pipeline import (  # noqa: E402
    fig_feature_importance,
    fig_pred_vs_actual,
    train_all,
    predict_from_sliders,
)
from src.theme import style_css  # noqa: E402

st.set_page_config(
    page_title="Smart Building Sustainability AI",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def _fetch_data(path_str: str, max_rows: int | None = None) -> pd.DataFrame:
    return load_dataset(path_str, max_rows=max_rows)


@st.cache_resource(show_spinner=False)
def _train_optimized_models(df: pd.DataFrame):
    return train_all(df)


def main() -> None:
    st.sidebar.title("Sustainability AI")
    dark = st.sidebar.toggle("Dark styling", value=False)
    st.markdown(style_css(dark), unsafe_allow_html=True)

    data_path = st.sidebar.text_input("Dataset CSV path", value=str(DEFAULT_CSV))
    path = Path(data_path)
    if not path.is_file():
        st.error(f"File not found: {path}")
        st.stop()

    try:
        df = _fetch_data(str(path.resolve()))
    except Exception as e:
        st.exception(e)
        st.stop()

    bundle = _train_optimized_models(df)

    st.title("Smart Building Sustainability Dashboard")
    st.markdown("Adjust settings to see real-time trends and predictions")

    # Input sliders
    st.subheader("Building Parameters")
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        occupancy = st.slider("Occupancy (%)", 
            float(df["occupancy_rate_percent"].min()),
            float(df["occupancy_rate_percent"].max()),
            float(df["occupancy_rate_percent"].median()))
        temperature = st.slider("Indoor temp (°C)", 
            float(df["indoor_temperature_degc"].min()),
            float(df["indoor_temperature_degc"].max()),
            float(df["indoor_temperature_degc"].median()))
    
    with col_b:
        hvac = st.slider("HVAC (kWh)", 
            float(df["hvac_consumption_kwh"].min()),
            float(df["hvac_consumption_kwh"].max()),
            float(df["hvac_consumption_kwh"].median()))
        lighting = st.slider("Lighting (kWh)", 
            float(df["lighting_consumption_kwh"].min()),
            float(df["lighting_consumption_kwh"].max()),
            float(df["lighting_consumption_kwh"].median()))
    
    with col_c:
        building_size = st.slider("Building size (m²)", 
            float(df["building_size_m2"].min()),
            float(df["building_size_m2"].max()),
            float(df["building_size_m2"].median()))

    # Get predictions
    out = predict_from_sliders(bundle, occupancy, temperature, hvac, lighting, building_size)
    
    # Show predictions
    st.subheader("Predictions & Daily Financial Impact")
    pred_col1, pred_col2, pred_col3 = st.columns(3)
    pred_col1.metric("Predicted Energy (kWh)", f"{out['energy_kwh']:,.0f}")
    pred_col2.metric("Predicted Water (L)", f"{out['water_l']:,.0f}")
    pred_col3.metric("Predicted Sustainability", f"{out['sustainability']:.3f}")

    cost_col1, cost_col2, cost_col3 = st.columns(3)
    cost_col1.metric("Energy Cost (Est)", f"${out['energy_cost_usd']:,.2f}")
    cost_col2.metric("Water Cost (Est)", f"${out['water_cost_usd']:,.2f}")
    cost_col3.metric("Total Daily Cost", f"${out['total_cost_usd']:,.2f}")

    # Show trends
    st.subheader("Historical Trends")
    
    # Resample to daily averages for faster rendering
    df_chart = df.resample("D").mean()

    # Interactive Plotly Charts
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df_chart.index, y=df_chart["energy_consumption_kwh"], fill='tozeroy', line=dict(color="#1f77b4", width=2), name="Actual"))
    fig1.add_hline(y=out['energy_kwh'], line_dash="dash", line_color="red", annotation_text="Your scenario")
    fig1.update_layout(title="Energy Consumption Over Time (Daily Avg)", yaxis_title="Energy (kWh)", height=350, margin=dict(l=0, r=0, t=30, b=0), template="plotly_dark" if dark else "plotly_white")
    st.plotly_chart(fig1, use_container_width=True)
    
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df_chart.index, y=df_chart["water_usage_liters"], fill='tozeroy', line=dict(color="#2ca02c", width=2), name="Actual"))
    fig2.add_hline(y=out['water_l'], line_dash="dash", line_color="red", annotation_text="Your scenario")
    fig2.update_layout(title="Water Usage Over Time (Daily Avg)", yaxis_title="Water (L)", height=350, margin=dict(l=0, r=0, t=30, b=0), template="plotly_dark" if dark else "plotly_white")
    st.plotly_chart(fig2, use_container_width=True)
    
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=df_chart.index, y=df_chart["sustainability_score"], fill='tozeroy', line=dict(color="#9467bd", width=2), name="Actual"))
    fig3.add_hline(y=out['sustainability'], line_dash="dash", line_color="red", annotation_text="Your scenario")
    fig3.update_layout(title="Sustainability Score Over Time (Daily Avg)", yaxis_title="Score", height=350, margin=dict(l=0, r=0, t=30, b=0), template="plotly_dark" if dark else "plotly_white")
    st.plotly_chart(fig3, use_container_width=True)

    # Show alerts
    st.subheader("⚠️ System Alerts & Decisions")
    alerts = generate_decision_alerts(df, occupancy, temperature, hvac, lighting, building_size, out)
    
    if not alerts:
        st.success("🟢 **System Optimal** - No active alerts for these parameters.")
    else:
        for alert in alerts:
            # Choose the colored container based on severity
            if alert["type"] == "error":
                container = st.error(f"**{alert['title']}**")
            else:
                container = st.warning(f"**{alert['title']}**")
                
            # Display formatted evidence and decision
            container.markdown(f"**Evidence:** {alert['evidence']}")
            container.markdown(f"**Action Required:** {alert['decision']}")

    # Model insights
    st.subheader("Model Insights")
    
    tab1, tab2 = st.tabs(["Predicted vs Actual", "Feature Importance"])
    
    with tab1:
        st.markdown(
            "**Model Accuracy - How well predictions match real data**  \n"
            "The Accuracy percentage measures how closely the ML model matches real-world historical data."
        )
        col1, col2, col3 = st.columns(3)
        for col, title, key in zip(
            (col1, col2, col3),
            ("Energy", "Water", "Sustainability"),
            ("energy", "water", "sustainability"),
        ):
            with col:
                st.markdown(f"**{title} model**")
                fig = fig_pred_vs_actual(bundle.pred_vs_actual[key], f"{title} Model")
                st.pyplot(fig)
                m = bundle.metrics[key]
                # Convert the R2 variance score into an easy-to-read Accuracy Percentage
                st.metric(f"{title} Accuracy", f"{m['r2'] * 100:.1f}%")
    
    with tab2:
        st.markdown("**Feature Importance - Which factors influence predictions most**")
        choice = st.selectbox(
            "Select prediction target:",
            ["energy", "water", "sustainability"],
            format_func=lambda x: x.title(),
        )
        fig_i = fig_feature_importance(
            bundle.importance[choice],
            f"Top factors for {choice} prediction",
        )
        st.pyplot(fig_i)

    # Export Report
    st.markdown("---")
    st.subheader("📥 Export Management Reports")
    csv = bundle.predictions_download.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Full ESG Predictions (.csv)",
        data=csv,
        file_name='building_esg_historical_predictions.csv',
        mime='text/csv',
    )


if __name__ == "__main__":
    main()
