import pandas as pd
import numpy as np

from a_dataset_model import return_dataset, return_daily_dataset, return_x_y, return_xgb_model, train_test, model_accuracy, peak_model

dataset = return_daily_dataset()

threshold_kwh = 0.5

# def recommend_device_upgrades(df):
#     global threshold_kwh
#     # Find average usage per device
#     avg_usage = df.groupby('device_id')['power_kwh'].mean()
    
#     # Recommend upgrade for devices with high average usage
#     high_usage_devices = avg_usage[avg_usage > threshold_kwh]
#     if high_usage_devices.empty:
#         print("No devices require an upgrade.")
#         return
    
#     for device, usage in high_usage_devices.items():
#         print(f"-Consider upgrading device {device} (Avg: {usage:.2f} kWh/hour) to a more energy-efficient model.")

# recommend_device_upgrades(dataset)

def get_upgrade_recommendations(
    dataset: pd.DataFrame,
    threshold_kwh: float = 0.5
) -> pd.DataFrame:
    """
    Returns a DataFrame of devices whose avg kWh/hour exceeds the threshold.
    Columns: device_id, appliance, avg_kwh_per_hour
    """
    df = dataset.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    # compute hourly rates
    avg_df = (
        df.groupby(['device_id','appliance'])['power_kwh']
          .mean()
          .reset_index(name='avg_kwh_per_hour')
    )
    # filter
    return avg_df[avg_df['avg_kwh_per_hour'] > threshold_kwh]
