import pandas as pd
from datetime import timedelta
import holidays

from a_dataset_model import return_dataset, return_x_y, return_xgb_model, train_test

# 1. Load & clean data
dataset = return_dataset()

x, y = return_x_y(dataset)
x_train, x_test, y_train, y_test, scaler = train_test(x, y)
model = return_xgb_model(x_train, y_train)

y_pred = model.predict(x_test)

# 4. UK holidays (or switch to your locale)
indian_holidays = holidays.CountryHoliday('IN', observed=False)

# 5. Tariff‐rate lookup (must match training)
def get_tariff_rate(hour: int) -> float:
    if 22 <= hour or hour < 6:
        return 3.0    # Night
    elif 6 <= hour < 10:
        return 5.0    # Morning
    elif 10 <= hour < 17:
        return 4.0    # Day
    else:
        return 6.0    # Evening

# 6. Build feature rows for any timestamp
def make_feature_df(ts: pd.Timestamp) -> pd.DataFrame:
    devices = dataset[['device_id','appliance']].drop_duplicates().reset_index(drop=True)
    df = devices.copy()
    df['temperature'] = dataset['temperature'].iloc[-1]
    df['hour']        = ts.hour
    df['dayofweek']   = ts.weekday()
    df['is_weekend']  = int(ts.weekday() >= 5)
    df['is_holiday']  = int(ts.normalize() in indian_holidays)
    df['month']       = ts.month
    df['tariff_rate'] = df['hour'].apply(get_tariff_rate)
    df['year']        = ts.year
    df['day']         = ts.day
    return df

def return_feats(ts: pd.Timestamp) -> pd.DataFrame:
    feats = make_feature_df(ts)
    feat_cols = [
        'device_id','temperature','hour','dayofweek',
        'is_weekend','is_holiday','month','tariff_rate', 'year','day'
    ]
    X_input = feats[feat_cols]
    
    X_scaled = scaler.transform(X_input)
    preds = model.predict(X_scaled)
    
    feats['predicted_power_kwh'] = preds
    feats['timestamp'] = ts

    return feats

# 7. Generic predict function
def predict_next_hour() -> pd.DataFrame:
    last_ts = pd.to_datetime(dataset['timestamp'].max())
    target_ts = last_ts + timedelta(hours=1)
    df = return_feats(target_ts)

    print("\n=== Next Hour Predictions ===")
    print(df[['device_id', 'appliance', 'predicted_power_kwh']].to_string(index=False))

def predict_hourly_power_timestamp(ts_str: str) -> pd.DataFrame:

    # ts_str = input("Enter timestamp (YYYY-MM-DD HH:MM:SS): ").strip()
    # try:
    ts = pd.to_datetime(ts_str)
    # except:
    #     print("Invalid format. Use YYYY-MM-DD HH:MM:SS")
    #     exit(1)

    df = return_feats(ts)

    print(f"\n=== Predictions for {df['timestamp'].iloc[0]} ===")
    print(df[['device_id','appliance','predicted_power_kwh']].to_string(index=False))
    
# predict_hourly_power_timestamp()
# predict_next_hour()