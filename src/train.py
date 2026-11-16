
import joblib
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from xgboost import XGBRegressor

from src.data import load_data, clean_data, split_data
from src.features import prepare_features


# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_DIR
RESULTS_DIR = PROJECT_DIR


def evaluate_model(model, X_data, Y_data):
    """
    Evaluate a regression model.
    """

    predictions = model.predict(X_data)

    rmse = np.sqrt(
        mean_squared_error(Y_data, predictions)
    )

    mae = mean_absolute_error(
        Y_data,
        predictions
    )

    r2 = r2_score(
        Y_data,
        predictions
    )

    return predictions, rmse, mae, r2


def train_models(X_train, Y_train, X_val, Y_val):
    """
    Train and compare multiple regression models.
    """

    models = {}

    # Linear Regression
    model_lr = LinearRegression()

    model_lr.fit(X_train, Y_train)

    models["Linear Regression"] = model_lr

    # Decision Tree
    model_dt = DecisionTreeRegressor(
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )

    model_dt.fit(X_train, Y_train)

    models["Decision Tree"] = model_dt

    # Random Forest
    model_rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=3,
        max_features=1.0,
        random_state=42,
        n_jobs=-1
    )

    model_rf.fit(X_train, Y_train)

    models["Random Forest"] = model_rf

    # XGBoost
    model_xgb = XGBRegressor(
        n_estimators=1000,
        early_stopping_rounds=50,
        learning_rate=0.01,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective="reg:squarederror",
        eval_metric="rmse",
        n_jobs=-1
    )

    model_xgb.fit(
        X_train,
        Y_train,
        eval_set=[(X_val, Y_val)],
        verbose=False
    )

    models["XGBoost"] = model_xgb

    # Validation results
    validation_results = []

    for name, model in models.items():

        _, rmse, mae, r2 = evaluate_model(
            model,
            X_val,
            Y_val
        )

        validation_results.append({
            "Model": name,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2
        })

    results_df = pd.DataFrame(validation_results)

    results_df = results_df.sort_values(
        by="RMSE",
        ascending=True
    ).reset_index(drop=True)

    return models, results_df


def evaluate_test(models, X_test, Y_test):
    """
    Evaluate trained models on the test set.
    """

    test_results = []
    test_predictions = {}

    for name, model in models.items():

        predictions, rmse, mae, r2 = evaluate_model(
            model,
            X_test,
            Y_test
        )

        test_predictions[name] = predictions

        test_results.append({
            "Model": name,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2
        })

    test_results_df = pd.DataFrame(test_results)

    test_results_df = test_results_df.sort_values(
        by="RMSE",
        ascending=True
    ).reset_index(drop=True)

    return test_results_df, test_predictions


def save_final_model(model, feature_names):
    """
    Save the final XGBoost model and its feature names.
    """

    artifact = {
        "model": model,
        "feature_names": feature_names,
        "model_name": "XGBoost",
        "target": "Demand"
    }

    model_path = MODEL_DIR / "electricity_demand_xgb_prediction_model.pkl"

    joblib.dump(artifact, model_path)

    print("\nFinal model saved:", model_path)


def main():

    # Load and clean data
    df = load_data()
    df = clean_data(df)

    # Feature engineering
    X, Y = prepare_features(df)

    # Split data
    (
        X_train_sub,
        X_val,
        X_test,
        Y_train_sub,
        Y_val,
        Y_test,
        X_train,
        Y_train
    ) = split_data(X, Y)

    # Train models
    models, validation_results = train_models(
        X_train_sub,
        Y_train_sub,
        X_val,
        Y_val
    )

    print("\nValidation Results:")
    print(validation_results)

    # Test evaluation
    test_results, test_predictions = evaluate_test(
        models,
        X_test,
        Y_test
    )

    print("\nTest Results:")
    print(test_results)

    # Save validation and test results
    validation_results.to_csv(
        RESULTS_DIR / "electricity_demand_validation_results.csv",
        index=False
    )

    test_results.to_csv(
        RESULTS_DIR / "electricity_demand_test_results.csv",
        index=False
    )

    # Save test predictions
    comparison_df = pd.DataFrame(
        test_predictions,
        index=X_test.index
    )

    comparison_df.insert(
        0,
        "Actual_Demand",
        Y_test
    )

    comparison_df.to_csv(
        RESULTS_DIR / "electricity_demand_predictions.csv"
    )

    # Refit final XGBoost on all training data
    best_n_estimators = (
        models["XGBoost"].best_iteration + 1
    )

    final_model = XGBRegressor(
        n_estimators=best_n_estimators,
        learning_rate=0.01,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective="reg:squarederror",
        eval_metric="rmse",
        n_jobs=-1
    )

    final_model.fit(X_train, Y_train)

    # Save final model
    save_final_model(
        final_model,
        X_train.columns.tolist()
    )

    print("\nTraining pipeline completed successfully!")


if __name__ == "__main__":
    main()
