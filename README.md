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

Start the api:

```bash
doit start_api
```

Start the streamlit demo web application

```bash
doit start_streamlit
```

## Models

The application loads pre-trained models from the `model` directory. Currently, the following models are available:

- XGBoost Regressor
