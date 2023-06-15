# FastAPI Simpson Strong-Tie Predictive Model API

## Description

This is a simple FastAPI application that provides an API endpoint for making predictions using pre-trained models.

## Installation

1. Clone this repository to your local machine.

2. Install the required dependencies:

   ```bash
   python -m venv venv
   .\\venv\\Scripts\\activate
   pip install -r requirements.txt
   ```

## Start the application

To start the application, run the following commands:

Start the backend:

```bash
doit start_backend
```

Start the streamlit demo web application (Make sure the backend is running first):

```bash
doit start_frontend
```

## Models

The application loads pre-trained models from the `model` directory. Currently, the following models are available:

- CatBoost Regressor
- Decision Tree Regressor
- Gradient Boosting Regressor
- K-Nearest Regressor
- LightGBM Regressor
- XGBoost Regressor
- XGBoost Optimized Regressor
- Lasso
- Linear Regression
- Random Forest
- Stacking Regressor
- Support Vector Regressor
