import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def process_data_reduced(file_path):
    # Load dataset
    data = pd.read_csv(file_path)

    # 1. Drop some columns
    # 'snow' has only 0 values in this dataset
    data = data.drop(columns=['snow', 'snowdepth', 'holiday', 'precip', 'cloudcover'], errors='ignore')

    # 2. Target Encoding
    # Map 'high_bike_demand' to 1 and 'low_bike_demand' to 0
    target_map = {'low_bike_demand': 0, 'high_bike_demand': 1}
    if 'increase_stock' in data.columns:
        data['increase_stock'] = data['increase_stock'].map(target_map)

    def categorize_hour(hour):
        if 15 <= hour <= 19:
            return 'Peak'
        elif 8 <= hour <= 14:
            return 'Day'
        else:
            return 'Night'

    data['time_period'] = data['hour_of_day'].apply(categorize_hour)
    # Then use One-Hot Encoding (pd.get_dummies) on 'time_period'
    data = pd.get_dummies(data, columns=['time_period'], drop_first=True)

    # 4. Features for Saturday and Sunday only
    if 'day_of_week' in data.columns:
        data['is_saturday'] = (data['day_of_week'] == 5).astype(int)
        data['is_sunday'] = (data['day_of_week'] == 6).astype(int)
        data = data.drop(columns=['day_of_week'], errors='ignore')

    # 4. Scaling Numerical Features
    # Only scale continuous variables. Leave binary flags (0/1) alone.
    continuous_cols = [
        'temp', 'dew', 'humidity', 'windspeed', 'visibility'
    ]

    # Select only columns present in the data
    cols_to_scale = [c for c in continuous_cols if c in data.columns]

    if cols_to_scale:
        scaler = StandardScaler()
        data[cols_to_scale] = scaler.fit_transform(data[cols_to_scale])

    return data
