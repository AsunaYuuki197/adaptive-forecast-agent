import pandas as pd


def create_lag_features(df: pd.DataFrame, target_col="Close", window=7):
    df_engineered = df.copy()
    for i in range(1, window + 1):
        df_engineered[f'lag_{i}'] = df_engineered[target_col].shift(i)
    return df_engineered.dropna()
