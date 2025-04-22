import pandas as pd
from datetime import timedelta
import holidays

from a_dataset_model import return_daily_dataset, return_x_y, train_test, return_xgb_model

# 1. Load & clean the daily dataset
dataset = return_daily_dataset()  # should return a DataFrame with a 'timestamp' column of dates
dataset['timestamp'] = pd.to_datetime(dataset['timestamp'])

# 2. Split into features & target
X, y = return_x_y(dataset)

# 3. Train/test split & scaler
x_train, x_test, y_train, y_test, scaler = train_test(X, y)

# 4. Train the XGBoost model (returns only the model)
model = return_xgb_model(x_train, y_train)

# 5. Holiday calendar for India
indian_holidays = holidays.CountryHoliday('IN', observed=False)

# 6. Helper to compute daily tariff rate
def daily_tariff_rate() -> float:
    def tr(h):
        if 22 <= h or h < 6:
            return 3.0
        if 6 <= h < 10:
            return 5.0
        if 10 <= h < 17:
            return 4.0
        return 6.0
    return sum(tr(h) for h in range(24)) / 24

DAILY_TARIFF = daily_tariff_rate()

# 7. Build feature rows for a given date
def make_daily_feature_df(ts: pd.Timestamp) -> pd.DataFrame:
    devices = dataset[['device_id','appliance','appliance_type']].drop_duplicates().reset_index(drop=True)
    df = devices.copy()
    df['timestamp']   = ts.normalize()
    # use the last known temperature or overall mean
    df['temperature'] = dataset['temperature'].iloc[-1]
    df['hour']        = ts.hour
    df['dayofweek']   = ts.weekday()
    df['is_weekend']  = int(ts.weekday() >= 5)
    df['is_holiday']  = int(ts.normalize() in indian_holidays)
    df['month']       = ts.month
    df['tariff_rate'] = DAILY_TARIFF
    df['year']        = ts.year
    df['day']         = ts.day
    return df

def return_feats(ts: pd.Timestamp) -> pd.DataFrame:
    feats = make_daily_feature_df(ts)
    feat_cols = [
       'device_id', 'temperature', 'is_weekend', 'is_holiday', 'tariff_rate',
       'year', 'month', 'day', 'dayofweek'
    ]
    X_input = feats[feat_cols]
    
    X_scaled = scaler.transform(X_input)
    preds = model.predict(X_scaled)
    
    feats['predicted_power_kwh'] = preds
    feats['timestamp'] = ts

    return feats

# 8. Predict next‐day consumption
def predict_next_day() -> pd.DataFrame:
    last_date = dataset['timestamp'].max()
    target    = pd.to_datetime(last_date) + timedelta(days=1)

    df = return_feats(target)

    # print("\n=== Next Day Predictions ===")
    # print(df[['device_id', 'appliance', 'predicted_power_kwh']].to_string(index=False))

# 9. Predict for an arbitrary date string
def predict_for_date(date_str: str) -> pd.DataFrame:

    ts = pd.to_datetime(date_str).normalize()
    
    # 2. Build feature rows & predict
    df = return_feats(ts)
    
    # 3. Sum up for total
    total_kwh = df['predicted_power_kwh'].sum()
    
    # 4. Print results
    # print(f"\n=== Predictions for {date_str} ===")
    # print(df[['device_id', 'appliance', 'predicted_power_kwh']].to_string(index=False))
    # print(f"\n🔋 Total predicted consumption for {date_str}: {total_kwh:.2f} kWh\n")
    
    # 5. Return the DataFrame if you want to further inspect it
    return df

predict_next_day()
predict_for_date("2025-04-22")




# Your next‑day numbers (≈0.01–0.02 kWh per device) are actually in line with what your UK‑DALE data is telling you, once you dig into the raw daily sums:

# You dropped the fridge.
# The fridge is the one device that runs almost continuously and racks up several kWh each day. By excluding it, your “total predicted” is only summing those infrequently used loads (kettle, microwave, TV, etc.), each of which only draws a few watts-hours per day in the UK data.

# UK‑DALE’s usage profiles are tiny.

# UK households boil water maybe twice a day (~0.03 kWh each boil) → ~0.06 kWh/day.

# TV on‑time is low.

# Washing machine runs once or twice.
# So seeing ~0.015 kWh (15 Wh) for kettle or TV is realistic for that dataset.

# Partial‑day at start and end.
# Your data begins on 2012‑11‑09 at 22:28, so the first “day” only has two hours of kettle/microwave events in it, pushing the per‑day average even lower.

# If you want more “Indian‑like” numbers, you have two choices:

# Include the fridge in your daily aggregation and prediction. It’ll add ~1–2 kWh/day back into your total.

# Use an Indian usage profile (either simulate higher baseload or pull in an Indian household dataset).