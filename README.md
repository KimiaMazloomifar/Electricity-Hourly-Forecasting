# ⚡ Electricity Demand Forecasting

**An End-to-End Machine Learning Project**

An end-to-end machine learning project for forecasting hourly electricity demand using historical electricity consumption and weather data.

This project demonstrates the complete machine learning lifecycle, including data preparation, exploratory data analysis, feature engineering, model training, evaluation, model serialization, REST API development, Docker containerization, and cloud deployment.

<blockquote>
**Live API:** https://electricity-hourly-forecasting-2.onrender.com/

**Interactive API Documentation:** https://electricity-hourly-forecasting-2.onrender.com/docs

**Health Check:** https://electricity-hourly-forecasting-2.onrender.com/health
</blockquote>

---

## Table of Contents

- [1. Project Overview](#1-project-overview)
- [2. Problem Description](#2-problem-description)
- [3. Dataset](#3-dataset)
- [4. Project Architecture](#4-project-architecture)
- [5. Data Preparation](#5-data-preparation)
- [6. Exploratory Data Analysis](#6-exploratory-data-analysis)
- [7. Feature Engineering](#7-feature-engineering)
- [8. Model Training and Evaluation](#8-model-training-and-evaluation)
- [9. Final Model and Hyperparameters](#9-final-model-and-hyperparameters)
- [10. Model Serialization](#10-model-serialization)
- [11. Project Structure](#11-project-structure)
- [12. Installation and Setup](#12-installation-and-setup)
- [13. Running the Training Pipeline](#13-running-the-training-pipeline)
- [14. FastAPI Web Service](#14-fastapi-web-service)
- [15. Docker Deployment](#15-docker-deployment)
- [16. Cloud Deployment](#16-cloud-deployment)
- [17. Making Predictions](#17-making-predictions)
- [18. Technologies and Tools](#18-technologies-and-tools)
- [19. Limitations and Future Improvements](#19-limitations-and-future-improvements)
- [20. Project Checklist](#20-project-checklist)

---

## 1. Project Overview

Electricity demand forecasting plays an important role in modern energy management. Accurate predictions of electricity consumption help energy providers plan power generation, optimize electricity distribution, reduce operational costs, and maintain grid stability.

In this project, I developed a machine learning regression solution to predict hourly electricity demand using:

- Historical electricity consumption.
- Time-based information.
- Weather conditions.
- Lagged demand observations.
- Rolling statistical features.


### Main Objectives

The main objectives of this project were to:

1. Identify an interesting real-world machine learning problem.
2. Find and prepare a relevant dataset.
3. Perform exploratory data analysis to understand the data.
4. Engineer meaningful features for time-series prediction.
5. Train and compare multiple regression algorithms.
6. Evaluate model performance and select the best model.
7. Convert the notebook workflow into reusable Python modules.
8. Build a REST API for serving predictions.
9. Containerize the application using Docker.
10. Deploy the prediction service to the cloud.

---

## 2. Problem Description

Electricity consumption is influenced by several temporal and environmental factors.

Demand can change depending on:

- The hour of the day.
- The day of the week.
- Seasonal patterns.
- Weekends and weekdays.
- Temperature and humidity.
- Recent electricity consumption.

The objective is to predict electricity demand for a particular hour based on these variables.

### Business Use Cases

An electricity provider could use an hourly demand forecasting model to support:

| Use Case | Description |
|---|---|
| Power Generation Planning | Estimate future electricity requirements and plan generation accordingly. |
| Load Balancing | Help balance electricity supply and demand. |
| Resource Allocation | Support efficient allocation of energy resources. |
| Operational Planning | Improve short-term operational decisions. |
| Consumption Analysis | Understand temporal electricity consumption patterns. |

### Machine Learning Approach

This problem is formulated as a **supervised regression task**.

The model receives a set of engineered input features and predicts a continuous numerical value representing electricity demand.

$$
\hat{y} = f(X)
$$

Where:

- \(X\) represents the input features.
- \(f\) represents the trained regression model.
- \(\hat{y}\) represents the predicted electricity demand.

---

## 3. Dataset

The project uses an hourly electricity demand dataset containing historical electricity consumption and weather-related information.

### Dataset Characteristics

| Property | Description |
|---|---|
| Dataset type | Time Series |
| Frequency | Hourly |
| Approximate observations | 43,848 |
| Start date | January 8, 2020 |
| End date | December 31, 2024 |
| Target variable | `Demand` |
| Weather variables | `Temperature`, `Humidity` |
| Problem type | Regression |

### Target Variable

`Demand` is the target variable representing the electricity demand that the model attempts to predict.

### Input Features

The final model uses 13 engineered features:

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

---

## 4. Project Architecture

The project follows a modular architecture that separates data processing, feature engineering, model training, prediction, and API development.

### Main Components

| Component | Responsibility |
|---|---|
| `data.py` | Loading, cleaning, and preparing the dataset |
| `features.py` | Creating time-based and historical demand features |
| `train.py` | Training, evaluating, comparing, and exporting models |
| `predict.py` | Loading the trained model and generating predictions |
| `app.py` | Exposing the model through FastAPI endpoints |
| `Dockerfile` | Defining the application container environment |

---

## 5. Data Preparation

The raw dataset was processed to make it suitable for time-series analysis and machine learning.

The main preprocessing steps included:

### 5.1 Timestamp Processing

- Converted the timestamp column into a Pandas datetime format.
- Removed records containing invalid timestamps.
- Set the timestamp as the DataFrame index.
- Sorted observations chronologically.

### 5.2 Missing Value Handling

Missing values were investigated and handled according to the type of variable.

For weather-related features:

- Time-based interpolation was applied.
- Forward filling and backward filling were used to handle remaining missing values.

For the target variable:

- Observations with missing electricity demand values were removed.

### 5.3 Data Validation

After preprocessing:

- The chronological order of observations was maintained.
- Missing target values were removed.
- Feature engineering was performed on the cleaned dataset.
- Rows containing missing values created by lag and rolling calculations were removed.

---

## 6. Exploratory Data Analysis

Exploratory Data Analysis (EDA) was performed to understand the dataset, identify consumption patterns, and investigate relationships between electricity demand and other variables.

The main analysis areas included:

### 6.1 Demand Distribution

Investigating the distribution of electricity demand to understand its range, variability, and potential outliers.

### 6.2 Temporal Patterns

Analyzing electricity consumption across different time dimensions:

- Hourly demand patterns.
- Daily consumption patterns.
- Monthly demand variations.
- Seasonal changes.
- Yearly consumption trends.

### 6.3 Weekday vs. Weekend Analysis

Comparing electricity consumption during weekdays and weekends to identify differences in demand behavior.

### 6.4 Weather Relationships

Investigating the relationships between electricity demand and:

- Temperature.
- Humidity.

### 6.5 Correlation Analysis

Examining correlations between numerical features to understand their relationships and identify potentially useful predictors.

### 6.6 Feature Analysis

Investigating the relevance of temporal, weather-related, and historical demand features for predicting electricity consumption.

---

## 7. Feature Engineering

Feature engineering was an important part of this project because electricity demand has strong temporal characteristics.

The following features were created from the timestamp and historical demand data.

### 7.1 Time-Based Features

| Feature | Purpose |
|---|---|
| `hour` | Captures hourly consumption patterns |
| `dayofweek` | Captures differences between days of the week |
| `month` | Captures monthly and seasonal variations |
| `year` | Represents yearly changes |
| `dayofyear` | Captures position within the year |
| `season` | Represents season-related variations |
| `weekofyear` | Captures weekly patterns |
| `is_weekend` | Distinguishes weekends from weekdays |

7.2 Lag Features

Lag features use previous demand values to help the model identify recurring patterns.

24-hour lag:

$$
Demand_{lag24}(t) = Demand(t-24)
$$

This feature represents electricity demand at the same hour on the previous day.

7.3 Rolling Statistical Features

Rolling features capture recent changes in electricity demand.

24-hour rolling mean:

$$
RollingMean_{24}(t) =
\frac{1}{24}\sum_{i=1}^{24} Demand(t-i)
$$

24-hour rolling standard deviation:

$$
RollingStd_{24}(t) =
Std(Demand(t-24), \ldots, Demand(t-1))
$$

Before calculating these features, the demand series was shifted by one hour to ensure that the current target value was not included in the calculations.

---

## 8. Model Training and Evaluation

Multiple regression algorithms were trained and compared to identify the most suitable model for electricity demand prediction.

### 8.1 Models Evaluated

Four regression algorithms were selected:

| Model | Description |
|---|---|
| Linear Regression | A baseline linear regression model |
| Decision Tree Regressor | A tree-based model that captures nonlinear relationships |
| Random Forest Regressor | An ensemble of decision trees |
| XGBoost Regressor | A gradient boosting algorithm based on decision trees |

### 8.2 Train, Validation, and Test Split

Since the dataset represents a time series, a chronological splitting strategy was used instead of random shuffling.

The initial split:

- 80% training and validation data.
- 20% final test data.

The training portion was further divided into training and validation subsets.

The validation data was used during model development, while the test data was reserved for final performance evaluation.

### 8.3 Evaluation Metrics

The models were evaluated using three regression metrics.

**Root Mean Squared Error (RMSE)**

Measures the square root of the average squared prediction errors. Larger errors receive a higher penalty.

$$
RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}
$$

**Mean Absolute Error (MAE)**

Measures the average absolute difference between actual and predicted values.

$$
MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|
$$

**R² Score**

Measures how much of the variance in the target variable is explained by the model.

$$
R^2 = 1-\frac{\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}{\sum_{i=1}^{n}(y_i-\bar{y})^2}
$$

### 8.4 Model Comparison

The following results were obtained on the test dataset:

| Model | RMSE | MAE | R² |
|---|---:|---:|---:|
| Linear Regression | 279.95 | 220.52 | 0.9601 |
| Decision Tree | 242.59 | 188.52 | 0.9700 |
| Random Forest | 177.42 | 135.91 | 0.9840 |
| **XGBoost** | **146.39** | **112.80** | **0.9891** |

### 8.5 Model Selection

XGBoost achieved the best performance among the evaluated models.

It obtained:

- The lowest RMSE.
- The lowest MAE.
- The highest R² score.

Based on these evaluation results, XGBoost was selected as the final model.

---

## 9. Final Model and Hyperparameters

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

### Early Stopping

Early stopping was used during training with the validation set.

This allows the training process to stop when validation performance no longer improves for the specified number of boosting rounds.

The final model was then exported for use in the prediction service.

---

## 10. Model Serialization

The selected XGBoost model was saved using Joblib.

**Model artifact:**

`electricity_demand_xgb_prediction_model.pkl`

The serialized artifact contains:

- The trained XGBoost model.
- The expected feature names.
- The model name.
- The target variable name.

The prediction module loads this artifact and uses it to generate predictions without requiring the model to be trained again.

---

## 11. Project Structure

The project is organized into reusable Python modules, with the notebook, dataset, model artifact, API, and deployment configuration files located in the project root.

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
│
├── Electricity Demand Dataset.csv
│
├── electricity_demand_xgb_prediction_model.pkl
│
├── app.py
│
├── requirements.txt
│
├── Dockerfile
│
├── .dockerignore
│
├── .gitignore
│
└── README.md
```

### Source Modules

**`src/data.py`**

Responsible for loading the dataset, cleaning the data, and preparing the chronological data splits.

**`src/features.py`**

Contains the feature engineering logic used to generate time-based and historical demand features.

**`src/train.py`**

Implements the model training and evaluation workflow, compares different regression algorithms, selects the final model, and exports the model artifact.

**`src/predict.py`**

Loads the serialized model and provides the functionality required to generate predictions.

**`app.py`**

Implements the FastAPI application and exposes the trained model through HTTP endpoints.

---

## 12. Installation and Setup

### Prerequisites

Make sure the following tools are installed:

- Python 3.14 or a compatible Python version supported by the project dependencies.
- pip
- Git
- Docker Desktop or Docker Engine.

### 12.1 Clone the Repository

```bash
git clone https://github.com/KimiaMazloomifar/Electricity-Hourly-Forecasting.git
cd Electricity-Hourly-Demand-Forecasting
```

### 12.2 Create a Virtual Environment

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

### 12.3 Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 13. Running the Training Pipeline

The training workflow is implemented as a reusable Python module.

To execute the pipeline from the project root:

```bash
python -m src.train
```

This process performs the training and evaluation workflow and exports the final trained model artifact.

Make sure the dataset is available in the expected project directory before running the training pipeline.

The trained model is saved as:

```text
electricity_demand_xgb_prediction_model.pkl
```

---

## 14. FastAPI Web Service

FastAPI was used to expose the trained machine learning model through a REST API.

The API loads the serialized XGBoost model and provides endpoints for checking service status and generating electricity demand predictions.

### API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Returns API information and running status |
| `/health` | GET | Checks service health |
| `/predict` | POST | Generates an electricity demand prediction |
| `/docs` | GET | Interactive Swagger documentation |
<img width="1866" height="880" alt="image" src="https://github.com/user-attachments/assets/5bd7dc67-8e4e-4454-a188-0ddd41b3ecd4" />

### Run the API Locally

From the project root:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Open the API documentation:

```text
http://localhost:8000/docs
```

The Swagger interface allows users to test the prediction endpoint directly from a browser.

---

## 15. Docker Deployment

The application was containerized using Docker to provide a consistent execution environment and simplify deployment.

### Docker Image Components

The Docker image includes:

- Python runtime.
- Required Python dependencies.
- `libgomp1`, required by XGBoost.
- FastAPI application.
- Python source modules.
- Serialized XGBoost model artifact.

### 15.1 Build the Docker Image

From the project root:

```bash
docker build -t electricity-demand-api .
```

### 15.2 Run the Docker Container

The application listens on port `8000` inside the container.

The host machine also exposes port `8000`.

```bash
docker run -d -p 8000:8000 --name electricity-demand-container electricity-demand-api
```

### 15.3 Access the Containerized API

**Main API:**

```text
http://localhost:8000/
```

**Swagger Documentation:**

```text
http://localhost:8000/docs
```

**Health Check:**

```text
http://localhost:8000/health
```

### 15.4 Container Management

Check running containers:

```bash
docker ps
```

View application logs:

```bash
docker logs electricity-demand-container
```

Stop the container:

```bash
docker stop electricity-demand-container
```

Remove the container:

```bash
docker rm electricity-demand-container
```

---

## 16. Cloud Deployment

As an additional step beyond the core project requirements, the FastAPI application was deployed to the cloud using **Render**.

This makes the prediction service publicly accessible without requiring users to install Python, download the model, or run Docker locally.

### Live Application

| Resource | URL |
|---|---|
| Live API | https://electricity-hourly-forecasting-2.onrender.com/ |
| Swagger Documentation | https://electricity-hourly-forecasting-2.onrender.com/docs |
| Health Check | https://electricity-hourly-forecasting-2.onrender.com/health |

### Deployment Overview

The cloud deployment uses the application source code and trained model artifact to run the FastAPI service on Render.

The deployed service exposes the same API endpoints as the local application.

**Note:** Render's free web services may spin down after a period of inactivity. The first request after inactivity may take longer because the service needs to start again.

---

## 17. Making Predictions

The `/predict` endpoint accepts a JSON object containing the 13 engineered input features and returns the predicted electricity demand.

The input features must be prepared before sending the request.

### 17.1 Example Request

**Method:** `POST`

**Endpoint:**

```text
/predict
```

**Request Body:**

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

### 17.2 Example Response

```json
{
  "predicted_demand": 3582.507568359375,
  "model_name": "XGBoost"
}
```

The predicted value above is an example response from the API. Actual predictions depend on the supplied input features.

### 17.3 Test the Live API

The prediction endpoint can be tested using the interactive Swagger interface:

https://electricity-hourly-forecasting-2.onrender.com/docs

Steps:

1. Open the Swagger documentation.
2. Expand `POST /predict`.
3. Click **Try it out**.
4. Enter the input JSON.
5. Click **Execute**.
6. Inspect the prediction response.

---

## 18. Technologies and Tools

The project uses the following technologies and libraries:

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

---

## 19. Limitations and Future Improvements

Although the project achieved strong evaluation results, there are several opportunities for further development.

### Forecasting Improvements

- Experiment with additional time-series forecasting approaches.
- Investigate more extensive hyperparameter optimization.
- Add additional historical lag features, such as 48-hour and 168-hour lags.
- Investigate the impact of external weather forecasts on future electricity demand.
- Explore prediction intervals to quantify forecasting uncertainty.

### API Improvements

- Allow the API to accept timestamps and historical observations instead of requiring users to calculate all engineered features manually.
- Add more comprehensive request validation.
- Add automated API tests.
- Improve error handling and logging.

### MLOps Improvements

- Add automated testing and continuous integration/deployment.
- Monitor model performance after deployment.
- Monitor data drift.
- Develop a process for periodically retraining the model.
- Track model versions and experiments.

### Forecasting Setup and Evaluation Considerations

The current API expects 13 engineered features as input. In particular, lag and rolling features depend on historical demand observations.

The evaluation setup is chronological and uses observed historical demand to construct lag and rolling features.

Therefore, the reported test performance represents a historical one-step-ahead prediction setup, rather than an independent multi-day forecast without access to newly observed demand.

This distinction is important when considering how the model could be used in a real-world forecasting system.

---

## 20. Project Checklist

This project was developed to apply the concepts covered in the Machine Learning Zoomcamp.

| ML Zoomcamp Requirement | Implementation | Status |
|---|---|---|
| Choose an interesting problem and find a dataset | Hourly electricity demand forecasting | Completed |
| Describe the problem and explain how a model can be used | Problem definition and business use cases | Completed |
| Prepare the data and perform EDA | Data cleaning, visualization, and analysis | Completed |
| Analyze important features | Time-based, weather, lag, and rolling features | Completed |
| Train multiple models | Linear Regression, Decision Tree, Random Forest, XGBoost | Completed |
| Tune and select the best model | XGBoost configuration and validation-based early stopping | Completed |
| Export the notebook into a script | Modular Python source files | Completed |
| Put the model into a web service | FastAPI REST API | Completed |
| Deploy the service locally with Docker | Docker image and container | Completed |
| Deploy the service to the cloud | Render deployment | Completed |
