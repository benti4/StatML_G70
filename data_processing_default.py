import pandas as pd

def process_data(file_path):
    # Load dataset
    data = pd.read_csv(file_path)

    # Encode categorical variables
    categorical_cols = data.select_dtypes(include=['object']).columns
    # Simple encoding: convert categorical columns to category type and then to codes (if n categories > 2)
    for col in categorical_cols:
        if data[col].nunique() <= 2:
            data[col] = data[col].astype('category').cat.codes
        else:
            data[col] = data[col].astype('category').cat.codes

    # normalize numerical features
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
    for col in numerical_cols:
        std = data[col].std()
        if std != 0:
            data[col] = (data[col] - data[col].mean()) / std
        else:
            # If std is 0, all values are the same, so we can set them to 0
            data[col] = 0

    return data
