
import joblib
import pandas as pd

from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_DIR / "electricity_demand_xgb_prediction_model.pkl"
)


def load_model():
    """
    Load the trained XGBoost model and feature names.
    """

    artifact = joblib.load(MODEL_PATH)

    model = artifact["model"]
    feature_names = artifact["feature_names"]
    model_name = artifact["model_name"]

    return model, feature_names, model_name


def predict_demand(input_features):
    """
    Generate electricity demand prediction.
    """

    model, feature_names, model_name = load_model()

    input_df = pd.DataFrame(
        [input_features]
    )

    # Check required features
    missing_features = [
        feature
        for feature in feature_names
        if feature not in input_df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing features: {missing_features}"
        )

    # Keep the same feature order used during training
    input_df = input_df[feature_names]

    prediction = model.predict(input_df)

    return {
        "predicted_demand": float(prediction[0]),
        "model_name": model_name
    }
