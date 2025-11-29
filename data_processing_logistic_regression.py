import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np


def process_data_logistic_regression(file_path):
    # Load dataset
    data = pd.read_csv(file_path)

    # 1. Target Encoding
    target_map = {'low_bike_demand': 0, 'high_bike_demand': 1}
    if 'increase_stock' in data.columns:
        data['increase_stock'] = data['increase_stock'].map(target_map)

    # 2. Time of Day Bins
    def categorize_hour(hour):
        if 15 <= hour <= 19:
            return 'Peak'
        elif 8 <= hour <= 14:
            return 'Day'
        else:
            return 'Night'

    data['time_period'] = data['hour_of_day'].apply(categorize_hour)
    data = pd.get_dummies(data, columns=['time_period'], drop_first=True)

    # 3. Weekend Flags
    if 'day_of_week' in data.columns:
        data['is_saturday'] = (data['day_of_week'] == 5).astype(int)
        data['is_sunday'] = (data['day_of_week'] == 6).astype(int)

    data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
    data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)

    data['is_winter'] = data['month'].isin([12, 1, 2]).astype(int)

    # Drop correlated or noisy columns
    # Dropping 'dew' because it correlates 0.87 with 'temp'
    cols_to_drop = ['snow', 'snowdepth', 'holiday', 'dew', 'hour_of_day', 'day_of_week', 'month']
    data = data.drop(columns=cols_to_drop, errors='ignore')

    # 5. Scaling
    # Only scale remaining continuous variables
    continuous_cols = ['temp', 'windspeed', 'visibility', 'precip', 'cloudcover']
    scaler = StandardScaler()
    # Check if cols exist before scaling
    valid_cols = [c for c in continuous_cols if c in data.columns]
    if valid_cols:
        data[valid_cols] = scaler.fit_transform(data[valid_cols])

    return data
