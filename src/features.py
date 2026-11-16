
import pandas as pd


def create_time_features(df):
    """
    Extract time-related features from Timestamp.
    """

    df = df.copy()

    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month
    df["year"] = df.index.year
    df["dayofyear"] = df.index.dayofyear

    # Season / quarter
    df["season"] = df.index.quarter

    # Week of year
    df["weekofyear"] = (
        df.index.isocalendar().week.astype(int)
    )

    # Weekend indicator
    df["is_weekend"] = (
        df.index.dayofweek.isin([5, 6])
    ).astype(int)

    return df


def create_lag_features(df):
    """
    Create lag and rolling statistical features.
    """

    df = df.copy()

    # Demand from the same hour yesterday
    df["Demand_lag_24hr"] = (
        df["Demand"].shift(24)
    )

    # Previous 24-hour rolling mean
    df["demand_rolling_mean_24hr"] = (
        df["Demand"]
        .shift(1)
        .rolling(window=24)
        .mean()
    )

    # Previous 24-hour rolling standard deviation
    df["demand_rolling_std_24hr"] = (
        df["Demand"]
        .shift(1)
        .rolling(window=24)
        .std()
    )

    return df


def prepare_features(df):
    """
    Apply feature engineering and prepare X and Y.
    """

    # Time features
    df = create_time_features(df)

    # Lag features
    df = create_lag_features(df)

    # Remove rows with missing lag features
    df = df.dropna()

    # Sort data chronologically
    df = df.sort_index()

    # Define target
    Y = df["Demand"]

    # Define features
    X = df.drop(columns=["Demand"])

    # Convert all features to numeric
    X = X.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # Remove remaining missing values
    valid_rows = X.notna().all(axis=1) & Y.notna()

    X = X.loc[valid_rows]
    Y = Y.loc[valid_rows]

    print("\nFeature engineering completed!")

    print("Final dataset shape:", df.shape)
    print("Number of features:", X.shape[1])
    print("Feature names:", X.columns.tolist())

    return X, Y
