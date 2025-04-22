import pandas as pd
from datetime import timedelta
import holidays

from a_dataset_model import return_dataset, return_daily_dataset, return_x_y, return_xgb_model, train_test, model_accuracy, peak_model

hourly = return_dataset()
# print(hourly['appliance'].unique())

daily_df = return_daily_dataset()
# print(daily_df['appliance'].unique())
# print(daily_df.head(2000))

x, y = return_x_y(hourly)

# print(x.columns)
x_train, x_test, y_train, y_test, scaler = train_test(x, y)
model = peak_model(x_train, y_train)

model_accuracy(model, x_train, y_train)

model_xg = return_xgb_model(x_train, y_train)
print("xg")
model_accuracy(model_xg, x_train, y_train)

print("\n")
print("=== Daily Dataset ===")  

x, y = return_x_y(daily_df)
x_train, x_test, y_train, y_test, scaler = train_test(x, y)
model = peak_model(x_train, y_train)

model_xg = return_xgb_model(x_train, y_train)

model_accuracy(model, x_train, y_train)
print("xg")
model_accuracy(model_xg, x_train, y_train)

