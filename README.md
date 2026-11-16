# Electricity Demand Forecasting

An end-to-end Machine Learning project for forecasting hourly electricity demand using historical electricity consumption and weather data. This project covers the complete ML workflow, from data preparation and exploratory data analysis to model training, evaluation, API development, Docker deployment, and cloud deployment.

**Live API:** https://electricity-hourly-forecasting-2.onrender.com/  
**API Documentation (Swagger):** https://electricity-hourly-forecasting-2.onrender.com/docs

---

## Table of Contents

- [Project Overview](#project-overview)
- [Problem Description](#problem-description)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Feature Engineering](#feature-engineering)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Final Model](#final-model)
- [Project Structure](#project-structure)
- [Installation and Setup](#installation-and-setup)
- [Running the Training Pipeline](#running-the-training-pipeline)
- [Running the API Locally](#running-the-api-locally)
- [Docker Deployment](#docker-deployment)
- [Cloud Deployment](#cloud-deployment)
- [Making Predictions](#making-predictions)
- [Technologies Used](#technologies-used)
- [Limitations and Future Improvements](#limitations-and-future-improvements)

---

## Project Overview

Electricity demand forecasting is an important task in modern energy management. Accurate predictions of electricity consumption can help energy providers optimize power generation, manage electricity distribution, reduce operational costs, and maintain grid stability.

In this project, I developed a machine learning solution to predict hourly electricity demand using historical demand patterns, time-based features, and weather conditions.

The project follows an end-to-end machine learning workflow:

1. Selecting a real-world problem and obtaining a dataset.
2. Understanding and preparing the data.
3. Performing exploratory data analysis (EDA).
4. Engineering relevant features.
5. Training and comparing multiple regression models.
6. Evaluating and selecting the best-performing model.
7. Exporting the machine learning workflow into reusable Python scripts.
8. Developing a prediction API using FastAPI.
9. Containerizing and deploying the application locally using Docker.
10. Deploying the API to the cloud using Render.

## Problem Description

Electricity consumption changes over time due to several factors, including:

- Hour of the day
- Day of the week
- Seasonal patterns
- Weekends and weekdays
- Temperature and humidity
- Recent electricity consumption patterns

The goal of this project is to build a regression model that predicts hourly electricity demand based on these variables.

### How can this model be used?

An electricity provider could use this model to estimate near-future electricity demand and support:

- Electricity generation planning.
- Load balancing and grid management.
- Energy resource allocation.
- Operational planning.
- Identifying changes in consumption patterns.

The model uses historical demand information and weather conditions to estimate electricity demand for a given hour.

## Dataset

The project uses an hourly electricity demand dataset containing historical electricity consumption and weather-related information.

### Dataset characteristics

| Property | Description |
|---|---|
| Data type | Time series |
| Frequency | Hourly |
| Approximate number of observations | 43,676 |
| Start date | January 8, 2020 |
| End date | December 31, 2024 |
| Target variable | `Demand` |
| Main input variables | Time features, weather conditions, and historical demand features |

### Target variable

`Demand` represents the electricity demand that the model is trained to predict.

### Input variables

The final model uses the following 13 features:

| Feature | Description |
|---|---|
| `hour` | Hour of the day |
| `dayofweek` | Day of the week |
| `month` | Month of the year |
| `year` | Year |
| `dayofyear` | Day number within the year |
| `season` | Season-related feature derived from the quarter |
| `weekofyear` | Week number within the year |
| `is_weekend` | Indicates whether the day is a weekend |
| `Temperature` | Temperature |
| `Humidity` | Relative humidity |
| `Demand_lag_24hr` | Electricity demand 24 hours earlier |
| `demand_rolling_mean_24hr` | Rolling mean of demand over the previous 24 hours |
| `demand_rolling_std_24hr` | Rolling standard deviation of demand over the previous 24 hours |

## Project Workflow

### 1. Data Preparation

The initial dataset was processed to ensure that it was suitable for time-series analysis and machine learning.

The preparation steps included:

- Converting the timestamp column into a datetime format.
- Removing invalid timestamps.
- Setting the timestamp as the DataFrame index.
- Sorting observations chronologically.
- Handling missing weather values using time-based interpolation and forward/backward filling.
- Removing observations with missing target values.
- Creating additional time-based and historical demand features.
- Removing rows with missing values generated during feature engineering.

### 2. Exploratory Data Analysis (EDA)

Exploratory Data Analysis was performed to understand the dataset, identify patterns, and investigate relationships between electricity demand and other variables.

The analysis focused on:

- Understanding the distribution of electricity demand.
- Examining demand patterns over time.
- Investigating hourly, daily, monthly, and seasonal variations.
- Comparing weekday and weekend consumption patterns.
- Exploring the relationship between weather conditions and electricity demand.
- Examining correlations between numerical features.
- Identifying potentially important features for prediction.

### 3. Feature Engineering

Feature engineering was used to extract useful information from timestamps and historical demand observations.

The main transformations included:

- Extracting hour, weekday, month, year, and day of year.
- Creating a season-related feature.
- Identifying weekends.
- Generating a 24-hour lag feature.
- Calculating rolling mean and standard deviation over a 24-hour window.

To avoid using the current target value when calculating historical rolling statistics, the demand series was shifted by one time step before calculating the rolling features.

### 4. Train, Validation, and Test Split

Because the dataset represents a time series, observations were split chronologically rather than randomly.

The initial split used:

- 80% for training and validation.
- 20% for final testing.

The training portion was further divided into training and validation subsets.

The validation set was used during model development, while the test set was reserved for final performance evaluation.

## Model Training and Evaluation

Multiple regression algorithms were trained and compared to identify the most suitable model for electricity demand prediction.

### Models evaluated

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- XGBoost Regressor

### Evaluation metrics

The models were evaluated using three standard regression metrics:

**RMSE (Root Mean Squared Error)**

Measures the square root of the average squared prediction error. It penalizes larger errors more heavily.

**MAE (Mean Absolute Error)**

Measures the average absolute difference between actual and predicted values.

**R² Score**

Measures how much of the variance in the target variable is explained by the model.

### Model comparison

The following results were obtained on the test dataset:

| Model | RMSE | MAE | R² |
|---|---:|---:|---:|
| Linear Regression | 279.95 | 220.52 | 0.9601 |
| Decision Tree | 242.59 | 188.52 | 0.9700 |
| Random Forest | 177.42 | 135.91 | 0.9840 |
| **XGBoost** | **146.39** | **112.80** | **0.9891** |

### Results

XGBoost achieved the best performance among the evaluated models.

Compared with the other models, it achieved:

- The lowest RMSE.
- The lowest MAE.
- The highest R² score.

These results indicate that XGBoost was the most effective model among the tested algorithms for this dataset and evaluation setup.

## Final Model

**Selected model: XGBoost Regressor**

The XGBoost model was configured with the following main hyperparameters:

| Parameter | Value |
|---|---:|
| `n_estimators` | 1000 |
| `learning_rate` | 0.01 |
| `max_depth` | 6 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `random_state` | 42 |
| `objective` | `reg:squarederror` |
| `eval_metric` | `rmse` |
| `early_stopping_rounds` | 50 |

Early stopping was used with the validation set to help prevent unnecessary boosting iterations.

After model selection, the final model was exported using Joblib.

The saved artifact contains:

- The trained model.
- The expected feature names.
- The model name.
- The target variable name.

**Model artifact:**

`electricity_demand_xgb_prediction_model.pkl`

## Exporting the Notebook into Python Scripts

The machine learning workflow was organized into reusable Python modules to make the project easier to maintain and deploy.

The main responsibilities are separated into:

- `data.py`: Data loading and preparation.
- `features.py`: Feature engineering.
- `train.py`: Model training, evaluation, comparison, and model export.
- `predict.py`: Loading the saved model and generating predictions.

The FastAPI application is implemented separately in `app.py`.

## Project Structure

```text
Electricity-Hourly-Demand-Forecasting/
│
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── features.py
│   ├── train.py
│   └── predict.py
│
├── electricity_demand_forecasting.ipynb
├── Electricity Demand Dataset.csv
├── electricity_demand_xgb_prediction_model.pkl
├── app.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
└── README.md
```

## Installation and Setup

### Prerequisites

Make sure the following tools are installed:

- Python 3.14 or a compatible Python version supported by the project dependencies.
- pip
- Git
- Docker (for containerized deployment)

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Electricity-Hourly-Demand-Forecasting
```

### 2. Create a virtual environment

**Windows:**

```powershell
python -m venv .venv
.venv\Scripts\activate
```

**Linux/macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Running the Training Pipeline

To execute the training pipeline from the project root:

```bash
python -m src.train
```

This process performs the training and evaluation workflow and exports the final trained model artifact.

Make sure the dataset is available in the expected project directory before running the training pipeline.

## Running the API Locally

The project uses FastAPI to expose the trained model through HTTP endpoints.

Start the API with:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Once the server is running, the following endpoints are available:

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | API information and running status |
| `/health` | GET | Health check |
| `/predict` | POST | Generate an electricity demand prediction |
| `/docs` | GET | Interactive Swagger API documentation |

Open the following address in your browser:

```text
http://localhost:8000/docs
```

## Docker Deployment

The application was containerized using Docker to ensure consistent execution across different environments.

The Docker image includes:

- Python runtime.
- Required Python dependencies.
- System dependency required by XGBoost.
- FastAPI application.
- Source modules.
- Trained XGBoost model artifact.

### 1. Build the Docker image

From the project root:

```bash
docker build -t electricity-demand-api .
```

### 2. Run the Docker container

The application listens on port `8000` inside the container. The same port is exposed on the host machine.

```bash
docker run -d -p 8000:8000 --name electricity-demand-container electricity-demand-api
```

### 3. Access the containerized API

The API is now available at:

**Main API:**

```text
http://localhost:8000/
```

**Swagger documentation:**

```text
http://localhost:8000/docs
```

**Health check:**

```text
http://localhost:8000/health
```

### 4. Check the running container

```bash
docker ps
```

To view container logs:

```bash
docker logs electricity-demand-container
```

To stop the container:

```bash
docker stop electricity-demand-container
```

To remove the container:

```bash
docker rm electricity-demand-container
```

## Cloud Deployment

As an additional deployment step, the FastAPI application was deployed to the cloud using **Render**.

The application is publicly accessible through the following URLs:

**Live API:**

https://electricity-hourly-forecasting-2.onrender.com/

**Swagger Documentation:**

https://electricity-hourly-forecasting-2.onrender.com/docs

**Health Check:**

https://electricity-hourly-forecasting-2.onrender.com/health

The cloud deployment makes it possible to access the prediction service without running the application locally.

> Note: Render's free web services may spin down after a period of inactivity. The first request after inactivity may take longer because the service needs to start again.

## Making Predictions

The `/predict` endpoint accepts a JSON object containing the 13 engineered input features and returns the predicted electricity demand.

### Example request

**POST**

```text
/predict
```

**Request body:**

```json
{
  "hour": 14,
  "dayofweek": 2,
  "month": 6,
  "year": 2024,
  "dayofyear": 163,
  "season": 2,
  "weekofyear": 24,
  "is_weekend": 0,
  "Temperature": 25.5,
  "Humidity": 45.0,
  "Demand_lag_24hr": 1200.0,
  "demand_rolling_mean_24hr": 1180.0,
  "demand_rolling_std_24hr": 35.0
}
```

### Example response

```json
{
  "predicted_demand": 3582.507568359375,
  "model_name": "XGBoost"
}
```

The predicted value above is illustrative; actual output depends on the supplied input features.

You can test the endpoint directly through the live Swagger interface:

https://electricity-hourly-forecasting-2.onrender.com/docs

## Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Data Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Gradient Boosting | XGBoost |
| Model Serialization | Joblib |
| API Development | FastAPI |
| API Server | Uvicorn |
| Data Validation | Pydantic |
| Containerization | Docker |
| Cloud Deployment | Render |
| Version Control | Git, GitHub |
| Development Environment | Jupyter Notebook |

## Limitations and Future Improvements

Although the project achieved strong results, there are several opportunities for improvement:

- Experiment with additional time-series forecasting approaches.
- Perform more extensive hyperparameter optimization.
- Investigate the impact of weather forecasts on future electricity demand predictions.
- Add more historical lag features, such as 48-hour and 168-hour lags.
- Explore prediction intervals to quantify forecasting uncertainty.
- Improve the API to accept timestamps and historical observations instead of requiring users to calculate all engineered features themselves.
- Add automated testing and continuous integration/deployment.
- Monitor model performance and data drift after deployment.

### Forecasting Setup

The current API expects the 13 engineered features as input. In particular, lag and rolling features depend on historical demand observations.

The evaluation setup is chronological and uses observed historical demand to construct lag and rolling features. Therefore, the reported test performance represents a historical one-step-ahead prediction setup, rather than an independent multi-day forecast without access to newly observed demand.

---

## Project Completion Checklist

| ML Zoomcamp Requirement | Status |
|---|---|
| Choose an interesting problem and find a dataset | Completed |
| Describe the problem and explain how a model can be used | Completed |
| Prepare the data and perform EDA | Completed |
| Analyze and engineer important features | Completed |
| Train multiple models | Completed |
| Evaluate models and select the best-performing model | Completed |
| Export the notebook workflow into Python scripts | Completed |
| Develop a web service for the model | Completed |
| Deploy the service locally using Docker | Completed |
| Deploy the service to the cloud | Completed — Render |

---

**Author:** [Your Name]

**Project:** Electricity Demand Forecasting

**Course:** Machine Learning Zoomcamp
