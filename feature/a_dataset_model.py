import pandas as pd

def return_dataset()-> pd.DataFrame:
    # Load and preprocess the dataset
    dataset = pd.read_csv(r"C:\Users\Akash\Desktop\electricity3\house_1_hourly.csv")
    
    dataset = dataset.dropna(subset=['power_kwh'])

    # Remove duplicate rows
    dataset.drop_duplicates(inplace=True)

    # Convert timestamp to datetime
    dataset['timestamp'] = pd.to_datetime(dataset['timestamp'])

    # Extract time feature
    dataset['year'] = dataset['timestamp'].dt.year
    dataset['month'] = dataset['timestamp'].dt.month
    dataset['day'] = dataset['timestamp'].dt.day
    dataset['dayofweek'] = dataset['timestamp'].dt.dayofweek

    # Remove outliers using IQR method
    Q1 = dataset['power_kwh'].quantile(0.25)
    Q3 = dataset['power_kwh'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    dataset = dataset[(dataset['power_kwh'] >= lower_bound) & (dataset['power_kwh'] <= upper_bound)]

    # Remove fridge data
    dataset = dataset[dataset['appliance'] != 'fridge']

    return dataset

def return_daily_dataset() -> pd.DataFrame:

    hourly_df = return_dataset()

    df = hourly_df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    agg_dict = {
        'power_kwh':   'sum',
        'temperature': 'mean',
        'is_weekend':  'first',
        'is_holiday':  'first',
        'tariff_rate': 'mean',
        'year':        'first',
        'month':       'first',
        'day':         'first',
        'dayofweek':   'first'
    }
    
    daily = (
        df
        .groupby([
            pd.Grouper(key='timestamp', freq='D'),
            'appliance', 'device_id', 'appliance_type'
        ], as_index=False)
        .agg(agg_dict)
    )
    
    # daily = daily.rename(columns={'timestamp': 'date'})
    return daily

def return_x_y(dataset: pd.DataFrame):
    """
    Build X,y without mutating the original `dataset`.
    """
    # 1. Figure out which columns to drop
    to_drop = ['timestamp', 'appliance', 'appliance_type', 'day_name']
    # 2. And always drop the target too
    drop_all = [c for c in to_drop if c in dataset.columns] + ['power_kwh']

    # 3. Create X in one go (inplace=False by default)
    X = dataset.drop(columns=drop_all, errors='ignore')

    # 4. Y comes straight from the original
    y = dataset['power_kwh']

    return X, y

def train_test(x, y, test_size=0.2, random_state=42):
    """
    Split the dataset into train and test sets.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)  # Use transform, not fit_transform!

    return x_train, x_test, y_train, y_test, scaler

def return_xgb_model(x_train, y_train):
    from xgboost import XGBRegressor
    # 1. Define and train the model
    x_train, x_test, y_train, y_test, scaler = train_test(x_train, y_train)

    model = XGBRegressor(
        subsample=0.8,
        n_estimators=300,
        max_depth=7,
        learning_rate=0.05,
        gamma=0,
        colsample_bytree=0.8,
        random_state=42
    )

    # 2. Fit the model
    model.fit(x_train, y_train)

    return model

def peak_model(x_train, y_train):
    from sklearn.ensemble import RandomForestRegressor
    model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    model_rf.fit(x_train, y_train)

    return model_rf

def model_accuracy(model, x_train, y_train):
  
    from sklearn.metrics import mean_squared_error, r2_score
    import numpy as np

    x_train, x_test, y_train, y_test, scaler = train_test(x_train, y_train)

    # Predict on training and testing sets
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    # Training scores
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)

    # Testing scores
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

        # Print results
    print("=== Training Data ===")
    print("MSE:", train_mse)
    print("R² Score:", train_r2)

    print("\n=== Testing Data ===")
    print("MSE:", test_mse)
    print("R² Score:", test_r2)

    return {
        "train_rmse": np.sqrt(mean_squared_error(y_train, y_train_pred)),
        "train_r2":  r2_score(y_train, y_train_pred),
        "test_rmse":  np.sqrt(mean_squared_error(y_test,  y_test_pred)),
        "test_r2":   r2_score(y_test,  y_test_pred)
    }
    r
