
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict import load_model


# Initialize FastAPI
app = FastAPI(
    title="Electricity Demand Forecasting API",
    description=(
        "An API for hourly electricity demand "
        "prediction using an XGBoost model."
    ),
    version="1.0.0"
)


# Load trained model once when the application starts
model, feature_names, model_name = load_model()


# Input schema
class DemandFeatures(BaseModel):

    hour: int = Field(ge=0, le=23)

    dayofweek: int = Field(ge=0, le=6)

    month: int = Field(ge=1, le=12)

    year: int

    dayofyear: int = Field(ge=1, le=366)

    season: int = Field(ge=1, le=4)

    weekofyear: int = Field(ge=1, le=53)

    is_weekend: int = Field(ge=0, le=1)

    Temperature: float

    Humidity: float

    Demand_lag_24hr: float

    demand_rolling_mean_24hr: float

    demand_rolling_std_24hr: float = Field(ge=0)


# Home endpoint
@app.get("/")
def home():

    return {
        "message": "Electricity Demand Forecasting API",
        "status": "running",
        "model": model_name
    }


# Health check endpoint
@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


# Prediction endpoint
@app.post("/predict")
def predict(features: DemandFeatures):

    try:

        input_data = features.model_dump()

        # Keep the exact feature order used during training
        input_data = {
            name: input_data[name]
            for name in feature_names
        }

        import pandas as pd

        input_df = pd.DataFrame(
            [input_data],
            columns=feature_names
        )

        prediction = model.predict(input_df)

        return {
            "predicted_demand": float(prediction[0]),
            "model_name": model_name
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )
