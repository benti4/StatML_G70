import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def process_data(file_path):
    # Load dataset
    data = pd.read_csv(file_path)

    # 1. Drop constant columns
    # 'snow' has only 0 values in this dataset
    if 'snow' in data.columns and data['snow'].nunique() <= 1:
        data = data.drop(columns=['snow'])

    # 2. Target Encoding
    # Map 'high_bike_demand' to 1 and 'low_bike_demand' to 0
    target_map = {'low_bike_demand': 0, 'high_bike_demand': 1}
    if 'increase_stock' in data.columns:
        data['increase_stock'] = data['increase_stock'].map(target_map)

    # 3. Cyclical Encoding for Time Features
    # Converts time units into sin/cos pairs to capture cyclical nature
    def encode_cyclical(df, col, max_val):
        df[col + '_sin'] = np.sin(2 * np.pi * df[col] / max_val)
        df[col + '_cos'] = np.cos(2 * np.pi * df[col] / max_val)
        return df

    # Apply to Hour (0-23), Day of Week (0-6), and Month (1-12)
    if 'hour_of_day' in data.columns:
        data = encode_cyclical(data, 'hour_of_day', 24)
    if 'day_of_week' in data.columns:
        data = encode_cyclical(data, 'day_of_week', 7)
    if 'month' in data.columns:
        data = encode_cyclical(data, 'month', 12)

    # Drop original time columns as they are now encoded
    data = data.drop(columns=['hour_of_day', 'day_of_week', 'month'], errors='ignore')

    # 4. Scaling Numerical Features
    # Only scale continuous variables. Leave binary flags (0/1) alone.
    continuous_cols = [
        'temp', 'dew', 'humidity', 'precip',
        'snowdepth', 'windspeed', 'cloudcover', 'visibility'
    ]

    # Select only columns present in the data
    cols_to_scale = [c for c in continuous_cols if c in data.columns]

    if cols_to_scale:
        scaler = StandardScaler()
        data[cols_to_scale] = scaler.fit_transform(data[cols_to_scale])

    return data
