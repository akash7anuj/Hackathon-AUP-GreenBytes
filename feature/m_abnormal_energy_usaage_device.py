
"""
Detects abnormal daily energy usage per device by comparing
actual vs. model‐predicted consumption and flagging large deviations.
"""

import pandas as pd
from a_dataset_model import return_daily_dataset, return_x_y, train_test, return_xgb_model



def abnormal_energy_usage():
    """
    Loads the daily dataset, trains/tests a model, computes prediction errors,
    and returns a DataFrame marking days with abnormally high error.
    """
    # 1. Load daily data
    daily_df = return_daily_dataset()  # must include columns ['date','device_id','appliance','power_kwh', ...]
    
    # 2. Extract X (features) and y (target)
    X, y = return_x_y(daily_df)
    
    # 3. Split into train/test and get scaler
    x_train, x_test, y_train, y_test, scaler = train_test(X, y)
    
    # 4. Train or load the model
    model = return_xgb_model(x_train, y_train)
    
    # 5. Predict on test set
    X_scaled = scaler.transform(x_test)
    y_pred   = model.predict(X_scaled)
    x_test_df = pd.DataFrame(x_test, columns=X.columns)
    
    # 6. Build results DataFrame
    results = x_test_df.copy()
    results['actual_usage']    = y_test.reset_index(drop=True)
    results['predicted_usage'] = y_pred
    results['error']           = results['actual_usage'] - results['predicted_usage']
    results['abs_error']       = results['error'].abs()
    
    # 7. Compute anomaly threshold = mean + 3*std of abs errors
    thresh = results['abs_error'].mean() + 3 * results['abs_error'].std()
    results['is_abnormal'] = results['abs_error'] > thresh
    
    # 8. Reattach appliance names (if dropped by return_x_y)
    mapping = daily_df[['device_id','appliance']].drop_duplicates()
    results = results.merge(mapping, on='device_id', how='left')
    
    # Reorder columns for clarity
    cols = ['device_id','appliance','actual_usage','predicted_usage','abs_error','is_abnormal']
    return results[cols], thresh

if __name__ == "__main__":
    df_anom, threshold = abnormal_energy_usage()
    print(f"\nAbnormal usage threshold (kWh): {threshold:.3f}\n")
    print(df_anom.to_string(index=False))
