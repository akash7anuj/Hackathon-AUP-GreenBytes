from b_next_hour_prediction import return_feats as next_hour_feats
from c_next_day_prediction import return_feats as next_day_feats
from d_schedule_week         import generate_weekly_schedule
import plotly.express        as px
import pandas                as pd
from a_dataset_model         import return_dataset, return_daily_dataset
import streamlit as st
from c_next_day_prediction import predict_for_date

elif section == "Forecast Graph":
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


    st.subheader("📊 Next Day Forecast per Device")

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

    st.markdown("---")

    st.subheader("📅 Next 7‑Day Forecast per Device")

    try:
        # EDIT: determine reference date
        last_ts = pd.to_datetime(return_dataset()['timestamp'].max())
        future_dates = [last_ts + pd.Timedelta(days=i) for i in range(1, 8)]

        # EDIT: collect per‑device predictions for each day
        records = []
        for d in future_dates:
            df_day = next_day_feats(d)
            # each row has device_id, appliance, predicted_power_kwh
            for _, row in df_day.iterrows():
                records.append({
                    'device_id': row['device_id'],
                    'appliance': row['appliance'],
                    'pred_kwh':  row['predicted_power_kwh']
                })

        wk_df = pd.DataFrame(records)

        # EDIT: aggregate over the 7 days per appliance
        agg = (
            wk_df
            .groupby('appliance')['pred_kwh']
            .sum()
            .reset_index()
        )

        # EDIT: bar chart of total kWh over next week per appliance
        fig3 = px.bar(
            agg,
            x='appliance',
            y='pred_kwh',
            labels={'appliance':'Appliance','pred_kwh':'Total kWh'},
            title="Next 7‑Day Total Predicted Consumption per Appliance"
        )
        st.plotly_chart(fig3, use_container_width=True)

    except Exception as e:
        st.error(f"Error generating 7‑day forecast: {e}")





# streamlit run "C:\Users\Akash\Desktop\electricity3\feature\zzz.py"