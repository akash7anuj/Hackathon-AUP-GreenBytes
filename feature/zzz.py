from b_next_hour_prediction import return_feats as next_hour_feats
from c_next_day_prediction import return_feats as next_day_feats
from d_schedule_week         import generate_weekly_schedule
import plotly.express        as px
import pandas                as pd
from a_dataset_model         import return_dataset, return_daily_dataset
import streamlit as st

# elif section == "Forecast":

st.header("🔮 Consumption Forecasts")

# --- 1) Next‑Hour Bar Chart ---
st.subheader("Next Hour Forecast per Device")
# a) find the timestamp for one hour after your latest data
last_ts = pd.to_datetime(return_dataset()['timestamp'].max())
target_hour = last_ts + pd.Timedelta(hours=1)
# b) get per‑device predictions
nh_df = next_hour_feats(target_hour)
# c) plot
fig1 = px.bar(
    nh_df,
    x='appliance',
    y='predicted_power_kwh',
    labels={'appliance':'Appliance','predicted_power_kwh':'kWh'},
    title=f"Predicted kWh by Appliance for {target_hour}"
)
st.plotly_chart(fig1, use_container_width=True)

st.markdown("---")

# --- 2) Next‑Day Hourly Line Chart ---
st.subheader("Next Day Hourly Profile")
today_max = last_ts
target_day = today_max + pd.Timedelta(days=1)
# a) get per‑device, per‑hour predictions for next day
nd_df = next_day_feats(target_day)
nd_hourly = (
    nd_df
    .assign(hour=lambda df: df['timestamp'].dt.hour)
    .groupby('hour')['predicted_power_kwh']
    .sum()
    .reset_index(name='pred_kwh')
)
# b) fetch actual last‑24h usage by hour
hist = (
    return_dataset()
    .assign(timestamp=lambda df: pd.to_datetime(df['timestamp']))
    .loc[lambda df: df['timestamp'] >= today_max - pd.Timedelta(hours=23)]
    .assign(hour=lambda df: df['timestamp'].dt.hour)
    .groupby('hour')['power_kwh']
    .sum()
    .reset_index(name='actual_kwh')
)
# c) merge and plot
comp = pd.merge(nd_hourly, hist, on='hour', how='left').fillna(0)
fig2 = px.line(
    comp,
    x='hour',
    y=['actual_kwh','pred_kwh'],
    labels={'value':'kWh','hour':'Hour of Day','variable':'Series'},
    title=f"Actual vs Predicted Hourly kWh for {target_day.date()}"
)
st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# --- 3) 7‑Day Total Consumption Outlook ---
st.subheader("7‑Day Total Consumption Outlook")
# a) build next‑7‑day totals by summing per‑device predictions
last_day = pd.to_datetime(return_daily_dataset()['timestamp'].max()).normalize()
future_dates = [last_day + pd.Timedelta(days=i) for i in range(1, 8)]
records = []
for d in future_dates:
    feats = next_day_feats(d)
    total_kwh = feats['predicted_power_kwh'].sum()
    records.append({'date': d.date(), 'predicted_kwh': total_kwh})
daily_totals = pd.DataFrame(records)
# b) area chart
fig3 = px.area(
    daily_totals,
    x='date',
    y='predicted_kwh',
    labels={'predicted_kwh':'kWh','date':'Date'},
    title="Predicted Total kWh over Next 7 Days"
)
st.plotly_chart(fig3, use_container_width=True)

# streamlit run "C:\Users\Akash\Desktop\electricity3\feature\zzz.py"