import pandas as pd
from a_dataset_model import return_dataset

def find_wasteful_devices(dataset):
    # 0) Ensure we have an 'hour' column
    df = dataset.copy()
    if 'hour' not in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour

    # 1) Devices active at night (midnight to 5 AM)
    night_usage = df[df['hour'].between(0, 5)]
    night_energy = night_usage.groupby('device_id')['power_kwh'].sum()

    # 2) Devices with very low variance (always on but doing little)
    device_variance = df.groupby('device_id')['power_kwh'].std().fillna(0)

    # 3) Devices with very low total usage
    total_usage = df.groupby('device_id')['power_kwh'].sum()

    # 4) Compile candidates
    records = []
    for dev in df['device_id'].unique():
        std   = device_variance.get(dev, 0)
        night = night_energy.get(dev, 0)
        total = total_usage.get(dev, 0)

        if std < 0.01 and night > 0 and total < 5:
            # lookup the appliance name (first occurrence)
            appl = df.loc[df['device_id'] == dev, 'appliance'].iloc[0]
            records.append({
                'device_id': dev,
                'appliance': appl,
                'std_kwh':    std,
                'night_kwh':  night,
                'total_kwh':  total
            })

    result_df = pd.DataFrame(records)

    # 5) Print summary
    if result_df.empty:
        print("✅ No potentially wasteful devices found.")
    else:
        print("=== Potentially Wasteful Devices ===")
        for _, row in result_df.iterrows():
            print(f"- Device {row.device_id} ({row.appliance}) might be wasting energy "
                  f"(std: {row.std_kwh:.4f}, night usage: {row.night_kwh:.2f} kWh, total: {row.total_kwh:.2f} kWh)")

    # 6) Return the DataFrame for further use
    return result_df

# Usage:
# if __name__ == "__main__":
#     ds = return_dataset()
#     df_waste = find_wasteful_devices(ds)
    # Now df_waste has columns: device_id, appliance, std_kwh, night_kwh, total_kwh
