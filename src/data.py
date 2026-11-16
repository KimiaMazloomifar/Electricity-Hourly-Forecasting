
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# Project root directory
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Dataset path
DATA_PATH = PROJECT_DIR / "Electricity Demand Dataset.csv"


def load_data(data_path=DATA_PATH):
    """
    Load the electricity demand dataset.
    """

    df = pd.read_csv(data_path)

    print("Dataset loaded successfully!")
    print("Original shape:", df.shape)

    return df


def clean_data(df):
    """
    Clean and prepare the raw dataset.
    """

    df = df.copy()

    # Convert Timestamp to datetime
    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        errors="coerce"
    )

    # Remove invalid timestamps
    df = df.dropna(subset=["Timestamp"])

    # Set Timestamp as index
    df = df.set_index("Timestamp")

    # Sort data chronologically
    df = df.sort_index()

    # Remove rows containing only null values
    df = df.dropna(how="all")

    # Check required columns
    required_columns = [
        "Demand",
        "Temperature",
        "Humidity"
    ]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"Required column '{column}' is missing!"
            )

    # Convert numerical columns
    for column in required_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Handle missing weather data
    df[["Temperature", "Humidity"]] = (
        df[["Temperature", "Humidity"]]
        .interpolate(method="time")
        .ffill()
        .bfill()
    )

    # Remove rows with missing target values
    df = df.dropna(subset=["Demand"])

    print("Data cleaning completed!")
    print("Cleaned shape:", df.shape)

    return df


def split_data(X, Y, test_size=0.20, validation_size=0.20):
    """
    Split data chronologically into training,
    validation and test sets.
    """

    # First split: Training and Test
    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=test_size,
        shuffle=False
    )

    # Second split: Training and Validation
    val_size = int(len(X_train) * validation_size)

    X_train_sub = X_train.iloc[:-val_size]
    Y_train_sub = Y_train.iloc[:-val_size]

    X_val = X_train.iloc[-val_size:]
    Y_val = Y_train.iloc[-val_size:]

    print("\nData splitting completed!")

    print("Training shape:", X_train_sub.shape)
    print("Validation shape:", X_val.shape)
    print("Test shape:", X_test.shape)

    return (
        X_train_sub,
        X_val,
        X_test,
        Y_train_sub,
        Y_val,
        Y_test,
        X_train,
        Y_train
    )
