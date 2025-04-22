
import pandas as pd
from a_dataset_model import (
    return_dataset,          # hourly
    return_daily_dataset     # daily
)

dataset = return_dataset()

START_TS = "2013-03-10 00:00:00"
END_TS   = "2013-03-11 23:59:59"

def find_top_energy_consuming_devices(dataset):
    print("=== Top Energy Consuming Devices ===")
    total_consumption = dataset.groupby('appliance')['power_kwh'].sum().sort_values(ascending=False)
    
    # print(total_consumption.head(top_n))
    # print(total_consumption)

find_top_energy_consuming_devices(dataset)

def find_top_n_energy_consuming_devices(dataset, top_n):
    print("=== Top Energy Consuming Devices ===")
    total_consumption = dataset.groupby('appliance')['power_kwh'].sum().sort_values(ascending=False)
    
    # print(total_consumption.head(top_n))
    # print(total_consumption.head(top_n))

find_top_n_energy_consuming_devices(dataset, top_n=5)




# Step 4: Analyze energy consumption
def find_top_n_energy_consumers_manually(dataset, top_n,start_time, end_time):
    
    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)

    filtered_df = dataset[(dataset['timestamp'] >= start_time) & (dataset['timestamp'] <= end_time)]

    total_energy = filtered_df.groupby('appliance')['power_kwh'].sum().sort_values(ascending=False)
    # print(f"Top {top_n} devices by energy usage from {start_time} to {end_time}:\n")
    
    # print(total_energy.head(top_n))

find_top_n_energy_consumers_manually(dataset, top_n=5, start_time=START_TS, end_time=END_TS)


def find_top_energy_consumers_manually(dataset, start_time, end_time):
    
    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)

    filtered_df = dataset[(dataset['timestamp'] >= start_time) & (dataset['timestamp'] <= end_time)]

    total_energy = filtered_df.groupby('appliance')['power_kwh'].sum().sort_values(ascending=False)
    # print(f"Top devices by energy usage from {start_time} to {end_time}:\n")
    
    # print(total_energy)

find_top_energy_consumers_manually(dataset, start_time=START_TS, end_time=END_TS)