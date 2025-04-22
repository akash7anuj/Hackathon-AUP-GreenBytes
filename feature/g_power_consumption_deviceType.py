
import pandas as pd
from a_dataset_model import (
    return_dataset,          # hourly
    return_daily_dataset     # daily
)

dataset = return_dataset()

def power_consumption_deviceType(dataset):

    # Group by appliance_type and sum power_kwh
    power_by_type = dataset.groupby('appliance_type')['power_kwh'].sum().reset_index()

    # Sort by total power consumption (descending)
    power_by_type = power_by_type.sort_values(by='power_kwh', ascending=False)

    # Display result
    # print("⚡ Total Power Consumption by Appliance Type:")
    # print(power_by_type)

power_consumption_deviceType(dataset)

start_date = "2013-03-10"
end_date = "2013-03-11"

def power_coms_range_deviceType(dataset, start_time, end_time):
    mask = (dataset['timestamp'] >= start_date) & (dataset['timestamp'] <= end_date)
    df_filtered = dataset.loc[mask]

    # Group by appliance_type and sum power_kwh
    power_by_type = df_filtered.groupby('appliance_type')['power_kwh'].sum().reset_index()

    # Sort by consumption
    power_by_type = power_by_type.sort_values(by='power_kwh', ascending=False)

    # Print result
    # print(f"⚡ Power Consumption by Appliance Type from {start_date} to {end_date}:")
    # print(power_by_type)

power_coms_range_deviceType(dataset, start_date, end_date)

def dataset_start_end_date(dataset):
    start_date = dataset['timestamp'].min()
    end_date = dataset['timestamp'].max()
    print(f"Dataset Start Date: {start_date}")
    print(f"Dataset End Date: {end_date}")

dataset_start_end_date(dataset)

def chart_range_deviceType(dataset, start_date, end_date):
    mask = (dataset['timestamp'] >= start_date) & (dataset['timestamp'] <= end_date)
    df_filtered = dataset.loc[mask]

    # Group by appliance_type and sum power_kwh
    power_by_type = df_filtered.groupby('appliance_type')['power_kwh'].sum().reset_index()

    # Sort by consumption
    power_by_type = power_by_type.sort_values(by='power_kwh', ascending=False)

    import plotly.express as px
    import os

    fig = px.pie(
        power_by_type,
        values='power_kwh',
        names='appliance_type',
        title=f'Device Type-Level Power Consumption Breakdown ({start_date} to {end_date})',
        hole=0.3
    )

    # fig.update_traces(textinfo='percent+label')

    # # Create folder if not exists
    # os.makedirs('image', exist_ok=True)

    # # Save the image
    # fig.write_image("image/device_power_chart_range.png")

    # print("Chart successfully saved to image/device_power_chart_range.png")


chart_range_deviceType(dataset, start_date, end_date)

def chart_deviceType(dataset):

    import plotly.express as px
    import os
    # Group by appliance_type and sum power_kwh
    power_by_type = dataset.groupby('appliance_type')['power_kwh'].sum().reset_index()

    # Sort by consumption
    power_by_type = power_by_type.sort_values(by='power_kwh', ascending=False)

    # Create pie chart
    fig = px.pie(
        power_by_type,
        values='power_kwh',
        names='appliance_type',
        title='Device Type-Level Power Consumption Breakdown',
        hole=0.3
    )

    # fig.update_traces(textinfo='percent+label')

    # Create folder if not exists
    # os.makedirs('image', exist_ok=True)

    # Save the image
    # fig.write_image("image/device_power_chart.png")

    # print("Chart successfully saved to image/device_power_chart.png")

chart_deviceType(dataset)