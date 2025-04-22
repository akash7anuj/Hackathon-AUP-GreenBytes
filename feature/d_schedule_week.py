import pandas as pd
from datetime import timedelta
import holidays
from a_dataset_model import (
    return_daily_dataset,
    return_x_y,
    train_test,
    return_xgb_model
)

# ------------------------------
# Constants & Configuration
# ------------------------------
LOOKBACK_DAYS = 28
FORECAST_DAYS = 7
INDIA_HOLIDAYS = holidays.CountryHoliday("IN")
DAYS_ORDER = ["Monday", "Tuesday", "Wednesday",
              "Thursday", "Friday", "Saturday", "Sunday"]

HEAVY  = {"dishwasher", "washing_machine", "geyser", "iron"}
MEDIUM = {"kettle", "microwave", "coffee_machine"}

REASONING_MAP = {
    "coffee_machine":  "Used with morning tea/breakfast.",
    "kettle":          "Frequent use; early day hydration.",
    "microwave":       "Lunch reheat. Short use duration.",
    "dishwasher":      "After dinner. Off‑peak usage.",
    "washing_machine": "Clothes wash after weekend. Spread loads.",
    "geyser":          "Morning showers; off‑peak if possible.",
    "iron":            "Batch ironing reduces energy spikes.",
    "fridge":          "Always on. Defrost weekly for efficiency.",
    "tv":              "Family entertainment in evening.",
    "laptop":          "Work/study hours. Charge off‑peak.",
    "ipad_charger":    "Overnight charging using off‑peak power.",
    "living_room_lamp":"Ambient lighting. Use LED; auto‑off timer.",
}

# ------------------------------
# Helper Functions
# ------------------------------
def load_and_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw timestamped data into daily aggregations by appliance.
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['timestamp']).dt.floor('D')
    daily = (
        df.groupby(['date', 'appliance'])
          .agg({
              'power_kwh':   'sum',
              'temperature': 'mean',
              'is_weekend':  'first',
              'is_holiday':  'first'
          })
          .reset_index()
    )
    return daily


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lag/rolling features and one-hot encode day-of-week.
    """
    df = df.sort_values('date').set_index('date')
    # lag features
    for lag in range(1, LOOKBACK_DAYS + 1):
        df[f'lag_{lag}'] = df['power_kwh'].shift(lag)
    # rolling stats
    df['rolling_mean_7'] = df['power_kwh'].shift(1).rolling(7).mean()
    df['rolling_std_7']  = df['power_kwh'].shift(1).rolling(7).std().fillna(0)
    # day-of-week one-hot
    dow_ohe = pd.get_dummies(df.index.day_name(), prefix='dow')
    return pd.concat([df, dow_ohe], axis=1).dropna()


def compress_days(days: list) -> str:
    """
    Convert list of day names into compact ranges, e.g. Mon–Wed, Fri.
    """
    idx = {day: i for i, day in enumerate(DAYS_ORDER)}
    seq = sorted(idx[d] for d in days)
    if seq == list(range(7)):
        return 'Daily'
    ranges, start, prev = [], seq[0], seq[0]
    for i in seq[1:]:
        if i == prev + 1:
            prev = i
        else:
            ranges.append((start, prev))
            start = prev = i
    ranges.append((start, prev))
    parts = []
    for s, e in ranges:
        if s == e:
            parts.append(DAYS_ORDER[s][:3])
        else:
            parts.append(f"{DAYS_ORDER[s][:3]}–{DAYS_ORDER[e][:3]}")
    return ", ".join(parts)


# ------------------------------
# Core Scheduling Logic
# ------------------------------
# Train a single XGBoost model on full daily history
full_daily = load_and_aggregate(return_daily_dataset().rename(columns={'timestamp':'timestamp'}))
X_full, y_full = return_x_y(full_daily)
# x_tr, x_te, y_tr, y_te, _ = train_test(X_full, y_full)

X_full, y_full = return_x_y(full_daily)

# DROP THE DATETIME COLUMN
if 'date' in X_full.columns:
    X_full = X_full.drop(columns=['date'])

x_tr, x_te, y_tr, y_te, _ = train_test(X_full, y_full)

model_full = return_xgb_model(x_tr, y_tr)




def _make_schedule(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Internal: generate schedule DataFrame from a daily DataFrame.
    """
    total_days  = (daily_df['date'].max() - daily_df['date'].min()).days + 1
    total_weeks = total_days / 7
    # historical usage by day-of-week
    hist_usage = (
        daily_df.groupby(['appliance', daily_df['date'].dt.day_name()])
                ['power_kwh'].mean().unstack(fill_value=0)
    )
    # how many runs per week
    runs_per_week = (
        daily_df.groupby('appliance')['power_kwh']
                .apply(lambda x: max(1, round(x.gt(0).sum() / total_weeks)))
                .to_dict()
    )

    records = []
    for appl in daily_df['appliance'].unique():
        sub = daily_df[daily_df['appliance']==appl].copy()
        feats = create_features(sub)
        if feats.empty:
            # fallback: top days by historical average
            runs = runs_per_week.get(appl, 1)
            days = hist_usage.loc[appl].sort_values(ascending=False).index.tolist()[:runs]
            reasoning = 'Fallback heuristic (insufficient history)'
        else:
            reasoning = 'Model-based & tariff optimization'
            temp = feats.copy()
            preds = []
            for _ in range(FORECAST_DAYS):
                d = temp.index[-1] + timedelta(days=1)
                base = {f'lag_{lag}': temp['power_kwh'].iloc[-lag]
                        for lag in range(1, LOOKBACK_DAYS+1)}
                base.update({
                    'rolling_mean_7': temp['power_kwh'].iloc[-7:].mean(),
                    'rolling_std_7':  temp['power_kwh'].iloc[-7:].std(),
                    'temperature':    temp['temperature'].iloc[-1],
                    'is_weekend':     int(d.weekday()>=5),
                    'is_holiday':     int(d in INDIA_HOLIDAYS)
                })
                for day in DAYS_ORDER:
                    base[f'dow_{day}'] = int(d.day_name()==day)
                Xp = pd.DataFrame([base])[X_full.columns]
                yhat = model_full.predict(Xp)[0]
                preds.append((d, yhat))
                temp = pd.concat([temp, pd.Series({'power_kwh': yhat,
                                                  'temperature': base['temperature']},
                                                 name=d).to_frame().T])
            # choose run days where prediction >= median
            dfp = pd.DataFrame(preds, columns=['date','pred']).set_index('date')
            med = dfp['pred'].median()
            days = dfp[dfp['pred']>=med].index.day_name().tolist()
        # choose time slot
        if appl in HEAVY:
            slot = '9:00 PM – 10:00 PM'
        elif appl in MEDIUM:
            slot = '7:00 AM – 8:00 AM' if appl=='coffee_machine' else '6:30 AM – 8:30 AM'
        else:
            if appl=='tv': slot = '8:00 PM – 10:30 PM'
            elif 'charger' in appl: slot = '10:00 PM – 6:00 AM'
            else: slot = '6:30 PM – 10:30 PM'
        records.append({
            'Appliance':      appl.replace('_',' ').title(),
            'Preferred Days': compress_days(days),
            'Time Slot':      slot,
            'Reasoning':      REASONING_MAP.get(appl, reasoning)
        })
    return pd.DataFrame(records)

# --- Automatic schedule for next week after data end ---
last_day = full_daily['date'].max()
auto_df = full_daily.copy()
# feed _make_schedule full history
schedule_df = _make_schedule(auto_df)
# print("Auto schedule (next week after dataset end):")
# print(schedule_df)


def generate_weekly_schedule(start_date: pd.Timestamp) -> pd.DataFrame:
    """
    Print & return schedule for the week following `start_date` based on history up to it.
    """
    history = full_daily[full_daily['date'] <= start_date]
    custom = _make_schedule(history)
    # print(f"Custom schedule for week starting {start_date.date()}: ")
    # print(custom)
    return custom

# # --- If run as script, prompt user for custom date ---
# if __name__ == '__main__':
#     date_str = input("Enter start date (YYYY-MM-DD) for custom schedule (or press Enter to skip): ").strip()
#     if date_str:
#         try:
#             sd = pd.to_datetime(date_str)
#             generate_weekly_schedule(sd)
#         except Exception:
#             print("Invalid date. Use format YYYY-MM-DD.")
