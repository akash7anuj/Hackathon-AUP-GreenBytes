# app.py
import streamlit as st
import pandas as pd
from datetime import timedelta, datetime
import holidays
import plotly.express as px
# at top of app.py
import j_solar_benefit as solar_simulation
import plotly.express as px


# -- your existing modules --
from a_dataset_model import (
    return_dataset,
    return_daily_dataset,
    return_x_y,
    train_test,
    return_xgb_model,
    peak_model,
    model_accuracy
)
import b_next_hour_prediction as nh
import c_next_day_prediction as nd
import d_schedule_week as sw
import e_top_5_device_energy_usage as top5
import f_energy_wasted_device as waste
import g_power_consumption_deviceType as ptype
import h_power_cons_appliances as pappl
import i_peak_hour_prediction as peak
import l_off_peak as offpeak
import m_abnormal_energy_usaage_device as abnormal

from b_next_hour_prediction import return_feats as next_hour_feats
from c_next_day_prediction import return_feats as next_day_feats
from d_schedule_week         import generate_weekly_schedule

st.set_page_config(page_title="Energy Dashboard", layout="wide")

st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", [
    "Dataset Overview",
    "Consumption by Appliance",
    "Consumption by DeviceType",
    "Next Hour Prediction",
    "Next Day Prediction",
    "Top applinaces Energy usage",
    "Peak Hour Prediction",
    "Off‑Peak Recommendations",
    "Weekly Schedule",
    "Solar Benefits",
    "Device Upgrade Recommendations",
    "Wasteful Devices",
    "abnormal energy usage Appliance",
    "Forecast Graph"
])

# 1: Dataset Overview
if section == "Dataset Overview":
    st.header("🔍 Dataset Overview")

    # 1. Show raw samples
    df_hourly = return_dataset()
    st.subheader("Hourly Data Sample")
    st.dataframe(df_hourly.head(10))

    df_daily = return_daily_dataset()
    st.subheader("Daily Aggregated Data Sample")
    st.dataframe(df_daily.head(10))

    # 2. Prepare data & train
    X, y = return_x_y(df_daily)
    x_train, x_test, y_train, y_test, scaler = train_test(X, y)

    rf_model  = peak_model(x_train, y_train)
    xgb_model = return_xgb_model(x_train, y_train)

    # 3. Compute metrics
    rf_metrics  = model_accuracy(rf_model, X, y)
    xgb_metrics = model_accuracy(xgb_model, X, y)

    # 4. Display in tables
    st.subheader("📈 XGBoost Performance ")
    st.table(pd.DataFrame([xgb_metrics]))

    st.subheader("📊 Random Forest Performance")
    st.table(pd.DataFrame([rf_metrics]))

# 2: Next Hour Prediction
elif section == "Next Hour Prediction":
    st.header("🔮 Next Hour Prediction")

    # 1) Custom timestamp prediction
    st.subheader("Predict for a custom timestamp")
    default_ts = return_dataset()['timestamp'].max()
    ts_input = st.text_input(
        "Enter timestamp (YYYY-MM-DD HH:MM:SS):",
        value=str(default_ts)
    )
    try:
        ts = pd.to_datetime(ts_input)
        custom_feats = nh.return_feats(ts)
        st.markdown(f"**Predictions for {ts}:**")
        st.dataframe(
            custom_feats[['device_id','appliance','predicted_power_kwh']]
        )
    except Exception:
        st.error("Invalid format. Please use YYYY‑MM‑DD HH:MM:SS")

    st.markdown("---")

    # 2) Next‑hour prediction
    st.subheader("Auto Next Hour Prediction")
    next_ts = pd.to_datetime(default_ts) + timedelta(hours=1)
    next_feats = nh.return_feats(next_ts)
    st.markdown(f"**Predictions for {next_ts}:**")
    st.dataframe(
        next_feats[['device_id','appliance','predicted_power_kwh']]
    )


# 3: Next Day Prediction
elif section == "Next Day Prediction":
    st.header("🔮 Next Day Prediction")

    # --- 1) User‐selected date ---
    st.subheader("Predict for a specific date")
    # Default to tomorrow
    last_date = return_daily_dataset()['timestamp'].max()
    default_date = (last_date + pd.Timedelta(days=1)).date()
    date_selected = st.date_input(
        "Choose your date:",
        value=default_date
    )
    # Convert and predict
    dt = pd.to_datetime(date_selected)
    custom_feats = nd.return_feats(dt)
    st.markdown(f"**Predictions for {dt.date()}:**")
    st.dataframe(
        custom_feats[['device_id','appliance','predicted_power_kwh']]
    )
    total_custom = custom_feats['predicted_power_kwh'].sum()
    st.markdown(f"**Total predicted consumption:** {total_custom:.2f} kWh")

    st.markdown("---")

    # --- 2) Automatic next‐day ---
    st.subheader("Auto Next Day Prediction")
    target = last_date + pd.Timedelta(days=1)
    auto_feats = nd.return_feats(target)
    st.markdown(f"**Predictions for {target.date()}:**")
    st.dataframe(
        auto_feats[['device_id','appliance','predicted_power_kwh']]
    )
    total_auto = auto_feats['predicted_power_kwh'].sum()
    st.markdown(f"**Total predicted consumption:** {total_auto:.2f} kWh")

elif section == "Weekly Schedule":
    st.header("📅 Weekly Optimization Schedule")

    # 1) Load your daily data and compute date bounds
    df_daily = return_daily_dataset()
    df_daily['date'] = pd.to_datetime(df_daily['timestamp']).dt.floor('D')
    min_date = df_daily['date'].min().date()
    max_date = df_daily['date'].max().date()

    # 2) User date picker (within data range)
    st.subheader("🗓️ Custom Weekly Schedule")
    user_date = st.date_input(
        "Select week‑start date:",
        value=max_date,
        min_value=min_date,
        max_value=max_date
    )
    st.markdown(f"*Data available from {min_date} to {max_date}*")

    # 3) Generate & display custom schedule
    custom_schedule = sw.generate_weekly_schedule(pd.to_datetime(user_date))
    st.markdown(f"**Schedule for week starting {user_date}:**")
    st.dataframe(custom_schedule)

    st.markdown("---")

    # 4) Display the automatic next‑week schedule
    st.subheader("🔄 Auto Next‑Week Schedule")
    st.dataframe(sw.schedule_df)


elif section == "Top applinaces Energy usage":
    st.header("🏆 Energy Consumption Analysis")
    df = return_dataset()

    # 1) Overall consumption of every appliance
    st.subheader("All Appliances – Total Consumption")
    total_all = (
        df
        .groupby('appliance')['power_kwh']
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    st.dataframe(total_all)

    # 2) Top‑N overall
    st.subheader("Top‑N Appliances (Overall)")
    top_n = st.number_input("Select N for overall top‑N:", min_value=1, max_value=20, value=5)
    st.dataframe(total_all.head(top_n))

    # 3) Custom time‑window Top‑N
    st.subheader("Top‑N in a Custom Time Window")
    START_TS = "2013-03-10 00:00:00"
    END_TS   = "2013-03-11 23:59:59"
    start_ts = st.text_input("Start timestamp (YYYY-MM-DD HH:MM:SS):", value=START_TS)
    end_ts   = st.text_input("End   timestamp (YYYY-MM-DD HH:MM:SS):", value=END_TS)

    try:
        st_dt = pd.to_datetime(start_ts)
        en_dt = pd.to_datetime(end_ts)
        mask = (df['timestamp'] >= st_dt) & (df['timestamp'] <= en_dt)
        window_df = df.loc[mask]

        if window_df.empty:
            st.warning("⚠️ No data in that time window.")
        else:
            # 3a) Top‑N in window
            st.markdown(f"**Top‑{top_n} appliances from {st_dt} to {en_dt}:**")
            window_topn = (
                window_df
                .groupby('appliance')['power_kwh']
                .sum()
                .sort_values(ascending=False)
                .reset_index()
            )
            st.dataframe(window_topn.head(top_n))

            # 4) Full list in window
            st.markdown(f"**All appliances from {st_dt} to {en_dt}:**")
            st.dataframe(window_topn)

    except Exception:
        st.error("Invalid timestamp format. Please use YYYY‑MM‑DD HH:MM:SS.")


elif section == "Wasteful Devices":
    st.header("🔋 Potentially Wasteful Devices")

    # 1) Load & ensure proper datetime/hour column
    df = return_dataset().copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour

    # 2) Compute stats
    std   = df.groupby('device_id')['power_kwh'].std().fillna(0)
    night = (
        df[df['hour'].between(0, 5)]
        .groupby('device_id')['power_kwh']
        .sum()
    )
    total = df.groupby('device_id')['power_kwh'].sum()

    # 3) Identify wasteful
    records = []
    for dev in df['device_id'].unique():
        if std.get(dev, 0) < 0.01 and night.get(dev, 0) > 0 and total.get(dev, 0) < 5:
            appl = df.loc[df['device_id'] == dev, 'appliance'].iloc[0]
            records.append({
                "Device ID": dev,
                "Appliance": appl,
                "Std (kWh)": std[dev],
                "Night (kWh)": night.get(dev, 0),
                "Total (kWh)": total[dev]
            })

    waste_df = pd.DataFrame(records)

    # 4) Display results
    if waste_df.empty:
        st.success("✅ No potentially wasteful devices found.")
    else:
        st.dataframe(waste_df)


elif section == "Consumption by DeviceType":
    st.header("⚡ Consumption by Appliance Type")
    df = return_dataset().copy()
    # ensure timestamp is datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # ----- Overall consumption -----
    st.subheader("Overall Consumption")
    overall = (
        df
        .groupby('appliance_type')['power_kwh']
        .sum()
        .reset_index()
        .sort_values('power_kwh', ascending=False)
    )
    st.dataframe(overall)
    fig_all = px.pie(
        overall,
        values='power_kwh',
        names='appliance_type',
        title="Overall Type‑Level Consumption",
        hole=0.3
    )
    st.plotly_chart(fig_all, use_container_width=True)

    st.markdown("---")

    # ----- Custom time window -----
    st.subheader("Custom Time Window Consumption")
    min_ts = df['timestamp'].min()
    max_ts = df['timestamp'].max()
    st.markdown(f"*Data available from {min_ts} to {max_ts}*")

    start_input = st.text_input(
        "Start timestamp (YYYY-MM-DD HH:MM:SS):",
        value=str(min_ts)
    )
    end_input = st.text_input(
        "End   timestamp (YYYY-MM-DD HH:MM:SS):",
        value=str(max_ts)
    )

    try:
        start_ts = pd.to_datetime(start_input)
        end_ts   = pd.to_datetime(end_input)

        mask = (df['timestamp'] >= start_ts) & (df['timestamp'] <= end_ts)
        df_range = df.loc[mask]

        if df_range.empty:
            st.warning("⚠️ No data in that time window.")
        else:
            by_type_range = (
                df_range
                .groupby('appliance_type')['power_kwh']
                .sum()
                .reset_index()
                .sort_values('power_kwh', ascending=False)
            )
            st.markdown(f"**Consumption from {start_ts} to {end_ts}:**")
            st.dataframe(by_type_range)

            fig_range = px.pie(
                by_type_range,
                values='power_kwh',
                names='appliance_type',
                title=f"Type‑Level Consumption ({start_ts.date()} to {end_ts.date()})",
                hole=0.3
            )
            st.plotly_chart(fig_range, use_container_width=True)

    except Exception:
        st.error("Invalid timestamp format. Please use YYYY‑MM‑DD HH:MM:SS.")

elif section == "Consumption by Appliance":
    st.header("⚡ Consumption by Appliance")
    df = return_dataset().copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # 0) Show data range
    min_ts = df['timestamp'].min()
    max_ts = df['timestamp'].max()
    st.markdown(f"*Data covers from {min_ts} to {max_ts}*")

    # 1) Overall consumption
    st.subheader("Overall Appliance Consumption")
    overall = (
        df.groupby('appliance')['power_kwh']
          .sum()
          .reset_index()
          .sort_values('power_kwh', ascending=False)
    )
    st.dataframe(overall)
    fig_all = px.pie(
        overall,
        values='power_kwh',
        names='appliance',
        title="Overall Appliance‑Level Consumption",
        hole=0.3
    )
    st.plotly_chart(fig_all, use_container_width=True)

    st.markdown("---")

    # 2) Custom time‑window consumption
    st.subheader("Custom Time Window Consumption")
    start_input = st.text_input(
        "Start timestamp (YYYY-MM-DD HH:MM:SS):",
        value=str(min_ts)
    )
    end_input = st.text_input(
        "End   timestamp (YYYY-MM-DD HH:MM:SS):",
        value=str(max_ts)
    )

    try:
        start_ts = pd.to_datetime(start_input)
        end_ts   = pd.to_datetime(end_input)

        window = df[(df['timestamp'] >= start_ts) & (df['timestamp'] <= end_ts)]
        if window.empty:
            st.warning("⚠️ No data in that time window.")
        else:
            by_app_range = (
                window.groupby('appliance')['power_kwh']
                      .sum()
                      .reset_index()
                      .sort_values('power_kwh', ascending=False)
            )
            st.markdown(f"**Consumption from {start_ts} to {end_ts}:**")
            st.dataframe(by_app_range)

            fig_range = px.pie(
                by_app_range,
                values='power_kwh',
                names='appliance',
                title=f"Appliance‑Level Consumption ({start_ts.date()} to {end_ts.date()})",
                hole=0.3
            )
            st.plotly_chart(fig_range, use_container_width=True)

    except Exception:
        st.error("Invalid timestamp format. Please use YYYY‑MM‑DD HH:MM:SS.")


elif section == "Peak Hour Prediction":
    st.header("🔝 Peak Usage Hour Prediction")

    df = return_dataset().copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    dev_to_appl = df.drop_duplicates('device_id').set_index('device_id')['appliance']
    X, y = return_x_y(df)
    feature_cols = X.columns.tolist()
    x_tr, x_te, y_tr, y_te, scaler = train_test(X, y)
    model = peak_model(x_tr, y_tr)

    fut_date = st.date_input("Pick date for peak prediction", value=datetime.today())
    fut = pd.to_datetime(fut_date)

    rows = []
    for dev in df['device_id'].unique():
        for hr in range(24):
            rows.append({
                'device_id':   dev,
                'temperature': df['temperature'].mean(),
                'hour':        hr,
                'dayofweek':   fut.weekday(),
                'is_weekend':  int(fut.weekday() >= 5),
                'is_holiday':  int(fut in holidays.CountryHoliday("IN")),
                'month':       fut.month,
                'tariff_rate': nh.get_tariff_rate(hr),
                'year':        fut.year,
                'day':         fut.day
            })
    fut_df = pd.DataFrame(rows)

    # convert to numpy to avoid the feature‑names warning
    X_pred = fut_df[feature_cols].values
    fut_df['predicted_power_kwh'] = model.predict(X_pred)

    fut_df['appliance'] = fut_df['device_id'].map(dev_to_appl)
    peak_df = fut_df.loc[fut_df.groupby('device_id')['predicted_power_kwh'].idxmax()]

    st.subheader(f"Peak usage for {fut.date()}")
    st.dataframe(peak_df[['device_id','appliance','hour','predicted_power_kwh']])


elif section == "Solar Benefits":
    st.header("☀️ Solar Panel Benefit Simulation")

    # 1) Simulation parameters
    st.subheader("Simulation Parameters")
    solar_gen = st.number_input(
        "Daily solar generation (kWh):", min_value=0.0, value=15.0, step=0.5
    )
    cost_kwh = st.number_input(
        "Electricity cost (₹/kWh):", min_value=0.0, value=7.0, step=0.5
    )
    co2_kwh = st.number_input(
        "CO₂ avoided per kWh (kg):", min_value=0.0, value=0.92, step=0.01
    )

    # 2) Historical vs. specific date
    st.subheader("Historical vs. Future")
    use_future = st.checkbox("Simulate for a specific date?", False)
    target = None
    if use_future:
        target = st.date_input(
            "Pick a date:", value=pd.to_datetime("today").date()
        ).strftime("%Y-%m-%d")

    # 3) Run simulation
    df_daily = return_daily_dataset()
    sim_df = solar_simulation.simulate_solar_benefits(
        df_daily,
        daily_solar_gen_kwh=solar_gen,
        cost_per_kwh=cost_kwh,
        co2_per_kwh=co2_kwh,
        target_date=target
    )

    # 4) Summary
    st.subheader("Summary")
    total_usage  = sim_df['usage_kwh'].sum()
    total_offset = sim_df['offset_kwh'].sum()
    total_grid   = sim_df['grid_kwh'].sum()
    avg_pct      = sim_df['pct_covered'].mean()
    total_cost   = sim_df['cost_saved'].sum()
    total_co2    = sim_df['co2_saved'].sum()

    st.markdown(f"- **Total usage:** {total_usage:.2f} kWh")
    st.markdown(f"- **Solar offset:** {total_offset:.2f} kWh")
    st.markdown(f"- **Grid draw:** {total_grid:.2f} kWh")
    st.markdown(f"- **Avg % covered:** {avg_pct:.1f}%")
    st.markdown(f"- **Cost saved:** ₹{total_cost:.2f}")
    st.markdown(f"- **CO₂ avoided:** {total_co2:.2f} kg")

    st.markdown("---")

    # 5) Detailed table
    st.subheader("Daily Breakdown")
    st.dataframe(sim_df)

    # 6) Charts

    # --- EDIT: Use Streamlit's bar_chart for a quick grouped bar plot ---
    st.subheader("Solar Offset vs. Grid Draw")
    df_chart = sim_df.copy()
    df_chart['date'] = pd.to_datetime(df_chart['date'])                # EDIT: ensure datetime
    df_chart = df_chart.set_index('date')                              # EDIT: index by date
    st.bar_chart(df_chart[['offset_kwh','grid_kwh']])                  # EDIT: grouped bars

    # --- EDIT: Alternatively, a Plotly grouped bar via melting ---
    st.subheader("Solar vs. Grid (Plotly)")
    melted = sim_df.melt(
        id_vars='date',
        value_vars=['offset_kwh','grid_kwh'],
        var_name='Source',
        value_name='kWh'
    )                                                                  # EDIT: long format
    fig1 = px.bar(
        melted,
        x='date',
        y='kWh',
        color='Source',
        barmode='group',
        title="Solar Offset vs. Grid Draw (kWh)"
    )
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Daily Coverage Percentage")
    fig2 = px.line(
        sim_df.assign(date=pd.to_datetime(sim_df['date'])),            # EDIT: ensure datetime
        x='date',
        y='pct_covered',
        labels={'pct_covered':'% Covered','date':'Date'},
        title="Solar Self‑Consumption (%)"
    )
    st.plotly_chart(fig2, use_container_width=True)


elif section == "Device Upgrade Recommendations":
    st.header("🔧 Device Upgrade Recommendations")

    # 1) Load data
    df = return_dataset().copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 2) Let user choose a threshold
    threshold = st.slider(
        "Minimum average usage (kWh/hr) to recommend upgrade:",
        min_value=0.0, max_value=5.0, value=0.5, step=0.1
    )

    # 3) Compute average usage per device and map appliance names
    avg_df = (
        df.groupby(['device_id', 'appliance'])['power_kwh']
          .mean()
          .reset_index(name='avg_kwh_per_hour')
          .sort_values('avg_kwh_per_hour', ascending=False)
    )

    # 4) Show bar chart of all devices
    st.subheader("📊 Average Usage per Device")
    chart_df = avg_df.set_index('appliance')['avg_kwh_per_hour']
    st.bar_chart(chart_df)  # quick visualization

    # 5) Show full table
    st.subheader("Detailed Table")
    st.dataframe(avg_df)

    st.markdown("---")

    # 6) Filter high‑usage devices
    high_df = avg_df[avg_df['avg_kwh_per_hour'] > threshold]

    # 7) Display recommendations
    st.subheader("⚠️ Upgrade Recommendations")
    if high_df.empty:
        st.success("None of your devices exceed the threshold—no upgrades needed!")
    else:
        for _, row in high_df.iterrows():
            st.markdown(
                f"- **{row.appliance}** (ID: {row.device_id}): "
                f"averages **{row.avg_kwh_per_hour:.2f} kWh/hr**, "
                "consider swapping for a more energy‑efficient model."
            )


# Then, in your Streamlit sidebar navigation:
elif section == "Off‑Peak Recommendations":
    st.header("💡 Off‑Peak Usage Suggestions")

    # 1) Parameter: how many top devices?
    top_n = st.slider("Analyze top N devices by peak usage:", 
                      min_value=1, max_value=20, value=5)

    # 2) Prepare data
    df = return_dataset().copy()

    # 3) Compute suggestions
    suggestions = offpeak.get_offpeak_suggestions(df, top_n=top_n)

    # 4) Bar chart of peak‑hour usage
    st.subheader("🔝 Peak‑Hour Usage (Top Devices)")
    chart_series = suggestions.set_index('Appliance')['Peak Usage kWh per week']
    st.bar_chart(chart_series)

    # 5) Detailed recommendations
    st.subheader("📋 Detailed Recommendations")
    st.dataframe(suggestions)

    # 6) Human‑friendly bullets
    st.subheader("📝 Recommendations Summary")
    if suggestions.empty:
        st.success("All devices are already operating in off‑peak windows.")
    else:
        for _, row in suggestions.iterrows():
            # EDIT: use .get() to avoid KeyError if column still missing
            off_peak = row.get("Recommended Off-Peak Hours", "N/A")
            st.markdown(
                f"- **{row.Appliance}** (Device {row['Device ID']}): "
                f"uses **{row['Peak Usage kWh per week']:.1f} kWh** during peak hours; "
                f"shift to off‑peak hours ({off_peak}) "
                f"to save ~₹{row['Est. Weekly Saving (₹)']:.2f} per week."
            )


elif section == "abnormal energy usage Appliance":
    st.header("🚨 Abnormal Energy Usage Detection")

    # 1) Run the detection
    df_anom, default_thresh = abnormal.abnormal_energy_usage()

    # 2) Show the threshold used
    st.markdown(f"**Anomaly threshold:** mean + 3 × std of abs. errors = {default_thresh:.3f} kWh")

    # 3) How many anomalies?
    n_abn = df_anom['is_abnormal'].sum()
    st.markdown(f"**Detected anomalies:** {n_abn} days out of {len(df_anom)} samples")

    # 4) Table of anomalies
    st.subheader("⚠️ Anomalous Records")
    abn_df = df_anom[df_anom['is_abnormal']].copy()
    st.dataframe(
        abn_df[['device_id','appliance','actual_usage','predicted_usage','abs_error']]
        .rename(columns={
            'device_id':'Device ID',
            'appliance':'Appliance',
            'actual_usage':'Actual (kWh)',
            'predicted_usage':'Predicted (kWh)',
            'abs_error':'Abs Error (kWh)'
        })
    )

    # 5) Visualize error distribution
    st.subheader("🔍 Error Distribution")
    fig1 = px.histogram(
        df_anom,
        x='abs_error',
        nbins=50,
        title="Histogram of Absolute Errors"
    )
    fig1.add_vline(x=default_thresh, line_dash="dash", annotation_text="Threshold")
    st.plotly_chart(fig1, use_container_width=True)

    # 6) Highlight top offenders
    st.subheader("🏆 Top 5 Deviations")
    top5 = df_anom.nlargest(5, 'abs_error')
    st.bar_chart(
        top5.set_index('appliance')['abs_error'],
        height=300
    )




elif section == "Forecast Graph":
    st.header("🔮 Consumption Forecasts")

    # --- Next‑Hour Bar Chart ---
    st.subheader("Next Hour per Device")
    last_ts = pd.to_datetime(return_dataset()['timestamp'].max())
    nh_df = next_hour_feats(last_ts + pd.Timedelta(hours=1))
    bar_fig = px.bar(
        nh_df,
        x='appliance',
        y='predicted_power_kwh',
        labels={'predicted_power_kwh':'kWh','appliance':'Device'},
        title="Predicted kWh by Appliance for Next Hour"
    )
    st.plotly_chart(bar_fig, use_container_width=True)

    st.markdown("---")

    # --- Next‑Day Line Chart ---
    st.subheader("Next Day Hourly Profile")
    today_max = pd.to_datetime(return_dataset()['timestamp'].max())
    nd_df = next_day_feats(today_max + pd.Timedelta(days=1))
    # sum across devices per hour
    nd_hourly = (
        nd_df
        .groupby(nd_df['timestamp'].dt.hour)['predicted_power_kwh']
        .sum()
        .reset_index(name='pred_kwh')
    )
    # if you like, fetch today's actual last‐24h by hour
    hist = (
        return_dataset()
        .assign(timestamp=lambda df: pd.to_datetime(df['timestamp']))
        .loc[lambda df: df['timestamp'] >= today_max - pd.Timedelta(hours=23)]
        .assign(hour=lambda df: df['timestamp'].dt.hour)
        .groupby('hour')['power_kwh']
        .sum()
        .reset_index(name='actual_kwh')
    )

    df2 = pd.merge(nd_hourly, hist, on='hour', how='left').fillna(0)
    line_fig = px.line(
        df2,
        x='hour',
        y=['actual_kwh','pred_kwh'],
        labels={'value':'kWh','hour':'Hour of Day','variable':'Legend'},
        title="Actual vs Predicted Hourly kWh for Next Day"
    )
    st.plotly_chart(line_fig, use_container_width=True)

    st.markdown("---")

    # --- 7‑Day Outlook Area Chart ---
    st.subheader("7‑Day Total Consumption Outlook")
    # Use your weekly schedule outputs as proxy daily totals
    custom_start = pd.to_datetime(return_daily_dataset()['timestamp'].max()) + pd.Timedelta(days=1)
    wk_df = generate_weekly_schedule(custom_start)
    # assume schedule_df has a 'Pred' column for each day; if not, sum predicted values
    daily_totals = (
        wk_df
        .groupby('date')['predicted_power_kwh']
        .sum()
        .reset_index()
    )
    area_fig = px.area(
        daily_totals,
        x='date',
        y='predicted_power_kwh',
        labels={'predicted_power_kwh':'kWh','date':'Date'},
        title="Predicted Total kWh over Next 7 Days"
    )
    st.plotly_chart(area_fig, use_container_width=True)




# streamlit run "C:\Users\Akash\Desktop\electricity3\feature\app.py"