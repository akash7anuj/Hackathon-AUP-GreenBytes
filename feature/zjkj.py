import streamlit as st
import pandas as pd
import plotly.express as px
from c_next_day_prediction import predict_for_date

st.header("📊 Next Day Predicted Consumption")

# 1) Let user pick a date
date_str = st.text_input(
    "Enter a date to forecast (YYYY-MM-DD):",
    value=str(pd.to_datetime("today").date() + pd.Timedelta(days=1))
)
try:
    # 2) Get per‑device predictions
    df_pred = predict_for_date(date_str)
    # 3) Sum per appliance
    chart_df = (
        df_pred
        .groupby('appliance')['predicted_power_kwh']
        .sum()
        .reset_index()
    )

    # 4) Plotly bar chart
    fig = px.bar(
        chart_df,
        x='appliance',
        y='predicted_power_kwh',
        labels={'appliance':'Appliance', 'predicted_power_kwh':'kWh'},
        title=f"Predicted kWh by Appliance on {date_str}"
    )
    st.plotly_chart(fig, use_container_width=True)

except Exception:
    st.error("Could not parse the date or run the forecast. Please use YYYY‑MM‑DD.")
