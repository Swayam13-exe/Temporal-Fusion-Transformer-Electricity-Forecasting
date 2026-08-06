"""
Streamlit demo: interactive fan-chart visualization of TFT quantile forecasts.

Usage:
    streamlit run src/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import plotly.graph_objects as go

from src.forecasting_service import load_serving_artifacts, generate_forecast

st.set_page_config(page_title="Electricity Load Forecast", layout="wide")


@st.cache_resource(show_spinner="Loading model...")
def get_artifacts():
    return load_serving_artifacts("config.yaml")


def build_fan_chart(result: dict) -> go.Figure:
    hist = result["history"]
    quantiles = result["quantiles"]
    forecast = result["forecast"]
    horizon = result["horizon_hours"]

    hist_x = list(hist["time_idx"])
    hist_y = list(hist["load_kwh"])

    last_hist_x = hist_x[-1]
    forecast_x = [last_hist_x + h for h in range(1, horizon + 1)]

    q_idx = {q: i for i, q in enumerate(quantiles)}
    p10 = [forecast[h][q_idx[0.1]] for h in range(horizon)]
    p25 = [forecast[h][q_idx[0.25]] for h in range(horizon)]
    p50 = [forecast[h][q_idx[0.5]] for h in range(horizon)]
    p75 = [forecast[h][q_idx[0.75]] for h in range(horizon)]
    p90 = [forecast[h][q_idx[0.9]] for h in range(horizon)]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=hist_x, y=hist_y, mode="lines", name="Observed history",
        line=dict(color="#1f77b4"),
    ))

    fig.add_trace(go.Scatter(
        x=forecast_x + forecast_x[::-1], y=p90 + p10[::-1],
        fill="toself", fillcolor="rgba(255,127,14,0.15)",
        line=dict(color="rgba(255,255,255,0)"), name="p10-p90", showlegend=True,
    ))

    fig.add_trace(go.Scatter(
        x=forecast_x + forecast_x[::-1], y=p75 + p25[::-1],
        fill="toself", fillcolor="rgba(255,127,14,0.30)",
        line=dict(color="rgba(255,255,255,0)"), name="p25-p75", showlegend=True,
    ))

    fig.add_trace(go.Scatter(
        x=forecast_x, y=p50, mode="lines+markers", name="Median forecast (p50)",
        line=dict(color="#ff7f0e"),
    ))

    fig.update_layout(
        xaxis_title="Hour (relative index)",
        yaxis_title="Load (kWh)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=40, b=10),
        height=500,
    )
    return fig


def main():
    st.title("Electricity Load Forecasting -- TFT Demo")
    st.caption(
        "24-hour probabilistic load forecast from a trained Temporal Fusion Transformer. "
        "This demo serves a small bundled sample of clients, not the full training dataset."
    )

    try:
        artifacts = get_artifacts()
    except RuntimeError as e:
        st.error(str(e))
        st.info("Run `python -m src.export_serving_artifacts --config config.yaml` first.")
        return

    clients = sorted(artifacts["demo_data"]["client_id"].unique().tolist())

    col1, col2 = st.columns([3, 1])
    with col1:
        client_id = st.selectbox("Select a client", clients)
    with col2:
        st.write("")
        st.write("")
        generate = st.button("Generate forecast", type="primary")

    if generate or "last_result" in st.session_state:
        if generate:
            with st.spinner("Running inference..."):
                try:
                    result = generate_forecast(client_id, artifacts)
                    st.session_state["last_result"] = result
                except ValueError as e:
                    st.error(str(e))
                    return

        result = st.session_state.get("last_result")
        if result and result["client_id"] == client_id:
            fig = build_fan_chart(result)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Median forecast, next 24 hours")
            median_idx = result["quantiles"].index(0.5)
            forecast_table = {
                "Hour ahead": list(range(1, result["horizon_hours"] + 1)),
                "Median forecast (kWh)": [
                    round(result["forecast"][h][median_idx], 1)
                    for h in range(result["horizon_hours"])
                ],
            }
            st.dataframe(forecast_table, use_container_width=True, hide_index=True)
        elif result:
            st.info("Client changed -- click 'Generate forecast' to update.")


if __name__ == "__main__":
    main()