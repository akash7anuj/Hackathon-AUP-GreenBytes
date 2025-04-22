import pandas as pd
import numpy as np

from a_dataset_model import return_dataset, return_daily_dataset, return_x_y, return_xgb_model, train_test, model_accuracy, peak_model

df = return_dataset()

# def suggest_off_peak_usage(df: pd.DataFrame, top_n: int = 5):
#     """
#     Analyze which appliances draw the most power during peak hours
#     and suggest shifting them to the cheapest off‐peak time windows.
#     """
#     print("=== Off‑Peak Usage Suggestions ===\n")

#     # Define peak and off‑peak hours
#     peak_hours = list(range(6, 10)) + list(range(18, 22))
#     all_hours = sorted(df['hour'].unique())
#     offpeak_hours = [h for h in all_hours if h not in peak_hours]

#     # Compute total consumption in peak hours per appliance
#     peak_df = df[df['hour'].isin(peak_hours)]
#     usage_by_device = (
#         peak_df
#         .groupby(['device_id', 'appliance'])['power_kwh']
#         .sum()
#         .sort_values(ascending=False)
#         .head(top_n)
#     )

#     # Compute average tariff by hour
#     tariff_by_hour = df.groupby('hour')['tariff_rate'].mean()
#     # Identify the 3 cheapest off‑peak hours
#     cheapest_hours = list(tariff_by_hour.sort_values().index[:3])

#     # Precompute average tariffs
#     avg_peak_tariff    = tariff_by_hour.loc[peak_hours].mean()
#     avg_offpeak_tariff = tariff_by_hour.loc[cheapest_hours].mean()

#     for (device_id, appliance), usage in usage_by_device.items():
#         # Estimate potential saving
#         saving_per_kwh = avg_peak_tariff - avg_offpeak_tariff
#         weekly_saving   = usage * saving_per_kwh

#         print(f"- {appliance.title()} (Device {device_id})")
#         print(f"  • Peak‑hour usage ({min(peak_hours)}–{max(peak_hours)}h): {usage:.2f} kWh/week")
#         print(f"  • Avg. tariff peak vs off‑peak: ₹{avg_peak_tariff:.2f}/kWh → ₹{avg_offpeak_tariff:.2f}/kWh")
#         print(f"  • Estimated weekly savings: ₹{weekly_saving:.2f}")
#         print(f"  • Recommended off‑peak hours: {', '.join(f'{h}:00' for h in cheapest_hours)}\n")

# # Example usage:
# suggest_off_peak_usage(dataset)


# At top of app.py (or in a utils module)
# import pandas as pd

def get_offpeak_suggestions(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    Returns a DataFrame of the top_n appliances by peak‑hour usage,
    with estimated savings and recommended off‑peak hours.
    """
    # ensure we have an hour column
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour

    # 1) Define peak & off‑peak
    peak_hours = list(range(6,10)) + list(range(18,22))
    tariff_by_hour = df.groupby('hour')['tariff_rate'].mean()

    # 2) Top‑N usage during peak
    peak_df = df[df['hour'].isin(peak_hours)]
    usage = (
        peak_df
        .groupby(['device_id','appliance'])['power_kwh']
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
    ).rename("peak_usage_kwh")

    # 3) Tariffs
    all_hours = sorted(df['hour'].unique())
    offpeak_hours = [h for h in all_hours if h not in peak_hours]
    cheapest_hours = list(tariff_by_hour.loc[offpeak_hours]
                          .sort_values()
                          .index[:3])
    avg_peak_tariff = tariff_by_hour.loc[peak_hours].mean()
    avg_offpeak_tariff = tariff_by_hour.loc[cheapest_hours].mean()

    # 4) Build suggestions
    records = []
    for (dev, appl), use_kwh in usage.items():
        saving_per_kwh = avg_peak_tariff - avg_offpeak_tariff
        weekly_saving   = use_kwh * saving_per_kwh
        records.append({
            "Device ID":               dev,
            "Appliance":               appl.title(),
            "Peak Usage kWh per week": use_kwh,
            "Peak Tarif (₹/kWh)":      avg_peak_tariff,
            "Off-Peak Tarif (₹/kWh)":  avg_offpeak_tariff,
            "Est. Weekly Saving (₹)":  weekly_saving,
            "Recommended Off-Peak Hours": 
                ", ".join(f"{h}:00" for h in cheapest_hours)
        })

    return pd.DataFrame(records)

