# solar_simulation.py
import pandas as pd
from a_dataset_model import return_daily_dataset
from c_next_day_prediction import return_feats  # uses your XGB daily predictor

def simulate_solar_benefits(
    df: pd.DataFrame,
    daily_solar_gen_kwh: float = 15.0,
    cost_per_kwh: float = 0.10,
    co2_per_kwh: float = 0.92,
    target_date: str = None
) -> pd.DataFrame:
    """
    If target_date is None: simulate for the last 7 days of history.
    If target_date is provided (YYYY-MM-DD):
      - if date in history, use actual usage
      - else, predict usage via return_feats()
    Returns DataFrame with:
      date, usage_kwh, solar_gen, offset_kwh, grid_kwh, pct_covered, cost_saved, co2_saved
    """
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date

    # build historical daily usage series
    daily_usage = df.groupby('date')['power_kwh'].sum()

    rows = []
    if target_date:
        dt = pd.to_datetime(target_date).date()
        if dt in daily_usage.index:
            usage = daily_usage.loc[dt]
        else:
            # predict next‐day usage
            feats = return_feats(pd.to_datetime(dt))
            usage = feats['predicted_power_kwh'].sum()
        rows = [(dt, usage)]
    else:
        # last up to 7 days
        last = daily_usage.tail(7).items()
        rows = list(last)

    # simulate
    records = []
    for date, usage in rows:
        solar  = daily_solar_gen_kwh
        offset = min(usage, solar)
        grid   = usage - offset
        pct    = (offset/usage*100) if usage>0 else 0
        cost   = offset * cost_per_kwh
        co2    = offset * co2_per_kwh
        records.append({
            'date':        date,
            'usage_kwh':   usage,
            'solar_gen':   solar,
            'offset_kwh':  offset,
            'grid_kwh':    grid,
            'pct_covered': pct,
            'cost_saved':  cost,
            'co2_saved':   co2
        })

    result = pd.DataFrame(records)
    # nice printout
    print("=== Solar Panel Simulation ===")
    if target_date:
        print(f"For {target_date}:")
    else:
        print("Last 7 days:")
    for r in records:
        print(f"{r['date']} | Usage: {r['usage_kwh']:.2f} kWh | "
              f"Solar: {r['offset_kwh']:.2f} kWh | Grid: {r['grid_kwh']:.2f} kWh | "
              f"{r['pct_covered']:.1f}% covered | ₹{r['cost_saved']:.2f} saved | "
              f"{r['co2_saved']:.2f} kg CO₂ avoided")

    return result

# # Example usage:
# if __name__=='__main__':
#     df_daily = return_daily_dataset()
#     # Historical:
#     simulate_solar_benefits(df_daily)
#     # Future:
#     simulate_solar_benefits(df_daily, target_date="2025-04-25")
