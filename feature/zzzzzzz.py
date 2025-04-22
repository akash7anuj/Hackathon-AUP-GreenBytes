from b_next_hour_prediction import return_feats as next_hour_feats
from c_next_day_prediction import return_feats as next_day_feats
from d_schedule_week         import generate_weekly_schedule
import plotly.express        as px
import pandas                as pd
from a_dataset_model         import return_dataset, return_daily_dataset
import streamlit as st

st.header("🔮 Consumption Forecasts")

# --- 1) Next‑Hour Bar Chart ---
st.subheader("Next Hour Forecast per Device")
last_ts = pd.to_datetime(return_dataset()['timestamp'].max())
nh_df = next_hour_feats(last_ts + pd.Timedelta(hours=1))
fig1 = px.bar(
    nh_df,
    x='appliance',
    y='predicted_power_kwh',
    labels={'appliance':'Appliance','predicted_power_kwh':'kWh'},
    title="Predicted kWh by Appliance for Next Hour"
)
st.plotly_chart(fig1, use_container_width=True)

st.markdown("---")

# --- 2) Next‑Day Hourly Line Chart ---
st.subheader("Next Day Hourly Profile")
today_max = last_ts
nd_df = next_day_feats(today_max + pd.Timedelta(days=1))
# aggregate per hour
nd_hourly = (
    nd_df
    .assign(hour=lambda df: df['timestamp'].dt.hour)
    .groupby('hour')['predicted_power_kwh']
    .sum()
    .reset_index(name='pred_kwh')
)
# actual last‑24h for comparison
hist = (
    return_dataset()
    .assign(timestamp=lambda df: pd.to_datetime(df['timestamp']))
    .loc[lambda df: df['timestamp'] >= today_max - pd.Timedelta(hours=23)]
    .assign(hour=lambda df: df['timestamp'].dt.hour)
    .groupby('hour')['power_kwh']
    .sum()
    .reset_index(name='actual_kwh')
)
comp = pd.merge(nd_hourly, hist, on='hour', how='left').fillna(0)
fig2 = px.line(
    comp,
    x='hour',
    y=['actual_kwh','pred_kwh'],
    labels={'value':'kWh','hour':'Hour of Day','variable':'Series'},
    title="Actual vs Predicted Hourly kWh for Next Day"
)
st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# --- 3) 7‑Day Total Outlook Area Chart ---
st.subheader("7‑Day Total Consumption Outlook")
last_day = pd.to_datetime(return_daily_dataset()['timestamp'].max())
wk_df = generate_weekly_schedule(last_day)
# sum predicted per date
daily_totals = (
    wk_df
    .groupby('date')['predicted_power_kwh']
    .sum()
    .reset_index()
)
fig3 = px.area(
    daily_totals,
    x='date',
    y='predicted_power_kwh',
    labels={'predicted_power_kwh':'kWh','date':'Date'},
    title="Predicted Total kWh over Next 7 Days"
)
st.plotly_chart(fig3, use_container_width=True)
