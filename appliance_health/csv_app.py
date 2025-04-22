# app.py
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.ensemble import IsolationForest

# Load Dataset
def load_data():
    df = pd.read_csv(r"C:\Users\Akash\Desktop\electricity2\house_4_daily.csv", parse_dates=['timestamp'])
    return df

# Feature Engineering
def extract_features(df):
    df['day'] = df['timestamp'].dt.date
    agg = df.groupby(['appliance', 'appliance_type', 'day']).agg({
        'power_kwh': ['mean', 'std', 'max', 'min'],
        'temperature': 'mean',
        'is_weekend': 'first',
        'is_holiday': 'first'
    }).reset_index()
    agg.columns = ['appliance', 'appliance_type', 'day', 'mean_power', 'std_power', 'max_power', 'min_power', 'temperature', 'is_weekend', 'is_holiday']
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
    st.set_page_config(page_title="Smart Appliance Health Dashboard", layout="wide")
    st.title("\U0001F9EA Appliance Health Monitoring")

    df = load_data()
    agg = extract_features(df)
    result = detect_anomalies(agg)

    st.subheader("📊 Daily Appliance Power Usage")
    fig = px.bar(result, x='day', y='mean_power', color='status',
                 barmode='group', facet_col='appliance',
                 title="Mean Daily Power Consumption by Appliance with Anomaly Detection")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("⚠️ Faulty Appliance Detection")
    faulty = result[result['status'] == 'Potential Fault']
    if not faulty.empty:
        st.warning("Potential faults detected in the following appliances:")
        st.dataframe(faulty[['appliance', 'appliance_type', 'day', 'mean_power', 'status']])
    else:
        st.success("All appliances are functioning normally.")

    st.subheader("📁 Raw Data Preview")
    st.dataframe(df.head())

if __name__ == '__main__':
    main()


# streamlit run "C:\Users\Akash\Desktop\electricity3\appliance_health\csv_app.py"