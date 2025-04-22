import streamlit as st

from a_dataset_model import return_dataset, return_daily_dataset
from b_next_hour_prediction import return_feats as next_hour_feats  # EDIT: use hourly predictor

# --- 3) 7‑Day Total Consumption Forecast ---
st.subheader("📆 7‑Day Total Consumption Forecast")

# a) Find last historical date
last_day = pd.to_datetime(return_daily_dataset()['timestamp'].max()).normalize()

# b) Build next‑7‑day totals via your daily predictor
future_dates = [last_day + pd.Timedelta(days=i) for i in range(1, 8)]
records = []
for d in future_dates:
    df7 = next_day_feats(d)                          # EDIT: use daily predictor
    total_kwh = df7['predicted_power_kwh'].sum()     # sum across all devices
    records.append({'date': d.date(), 'total_kwh': total_kwh})

weekly_df = pd.DataFrame(records)

# c) Plot a bar chart
fig_wk = px.bar(
    weekly_df,
    x='date',
    y='total_kwh',
    labels={'date':'Date','total_kwh':'Predicted kWh'},
    title="Predicted Total Consumption Over the Next 7 Days"
)
st.plotly_chart(fig_wk, use_container_width=True)
