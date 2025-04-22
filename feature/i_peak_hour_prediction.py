from a_dataset_model import return_dataset, return_x_y, return_xgb_model, train_test, peak_model
import pandas as pd
from b_next_hour_prediction import make_feature_df, get_tariff_rate
# 1. Load & clean data
dataset = return_dataset()

x, y = return_x_y(dataset)
x_train, x_test, y_train, y_test, scaler = train_test(x, y)

feature_cols = x.columns.tolist()

model = peak_model(x_train, y_train)

y_pred = model.predict(x_test)


def peak_usage_predict(dataset, model):
    # ========== STEP 5: Predict Future Peak Hour on a Given Date ==========
    future_date = pd.to_datetime("2025-04-18")

    is_weekend = 1 if future_date.weekday() >= 5 else 0
    temperature = dataset['temperature'].mean()
    is_holiday = 0  # Adjust based on your holiday calendar

    future_features = []
    for device_id in dataset['device_id'].unique():
        for hour in range(24):
            row = {
                'device_id': device_id,
                'temperature': temperature,
                'hour': hour,
                'dayofweek': future_date.weekday(),
                'is_weekend': is_weekend,
                'is_holiday': is_holiday,
                'month': future_date.month,
                'tariff_rate': get_tariff_rate(hour),
                'year': future_date.year,
                'day': future_date.day
            }
            future_features.append(row)

    df_future = pd.DataFrame(future_features)

    # 🔮 Predict energy usage for each hour
    df_future['predicted_power_kwh'] = model.predict(df_future)

    # 🔝 Find peak usage hour for each device
    peak_hours = df_future.loc[df_future.groupby('device_id')['predicted_power_kwh'].idxmax()]
    peak_hours['appliance'] = dataset['appliance'].unique()

    # print("\n🔮 Predicted Peak Usage Hour per Device on", future_date.date())
    # print(peak_hours[['device_id','appliance', 'hour', 'predicted_power_kwh']])


peak_usage_predict(dataset, model)