# app.py
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

# Simulate Appliance Power Usage Data
def simulate_data():
    np.random.seed(42)
    appliances = ['fridge', 'ac', 'washing_machine', 'tv']
    start_date = datetime(2025, 4, 1)
    data = []
    for appliance in appliances:
        for day in range(1, 18):
            mean = np.random.uniform(0.1, 1.0)
            for hour in range(24):
                timestamp = start_date + timedelta(days=day-1, hours=hour)
                power = np.random.normal(loc=mean, scale=0.05)
                # Inject anomaly
                if appliance == 'ac' and day == 10 and 12 <= hour <= 16:
                    power *= 3
                data.append([timestamp, appliance, max(power, 0.05)])
    return pd.DataFrame(data, columns=['timestamp', 'appliance', 'power_kwh'])

# Feature Engineering
def extract_features(df):
    df['day'] = df['timestamp'].dt.date
    agg = df.groupby(['appliance', 'day']).agg({
        'power_kwh': ['mean', 'std', 'max', 'min']
    }).reset_index()
    agg.columns = ['appliance', 'day', 'mean_power', 'std_power', 'max_power', 'min_power']
    return agg

# Anomaly Detection
def detect_anomalies(df):
    model = IsolationForest(contamination=0.05, random_state=42)
    features = ['mean_power', 'std_power', 'max_power', 'min_power']
    df['anomaly'] = model.fit_predict(df[features])
    df['status'] = df['anomaly'].map({-1: 'Potential Fault', 1: 'Normal'})
    return df

# Streamlit UI
def main():
    st.set_page_config(page_title="Appliance Health Monitoring", layout="wide")
    st.title("\U0001F9EA Appliance Health Monitoring Dashboard")

    df = simulate_data()
    agg = extract_features(df)
    result = detect_anomalies(agg)

    st.subheader("Aggregated Power Consumption")
    fig = px.bar(result, x='day', y='mean_power', color='status',
                 barmode='group', facet_col='appliance', title="Daily Power Consumption & Status")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Detected Faults")
    faulty = result[result['status'] == 'Potential Fault']
    if not faulty.empty:
        st.warning("⚠️ Faulty Appliances Detected:")
        st.dataframe(faulty[['appliance', 'day', 'mean_power', 'status']])
    else:
        st.success("✅ All appliances appear to be functioning normally.")

if __name__ == '__main__':
    main()


# streamlit run "C:\Users\Akash\Desktop\electricity3\appliance_health\appliance_health.py"
