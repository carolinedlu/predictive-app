from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI()

# Specify the directory where the models are stored
model_directory = "model"

# Load all models from the model directory
models = {}
for filename in os.listdir(model_directory):
    if filename.endswith(".joblib"):
        model_name = os.path.splitext(filename)[0]
        model_path = os.path.join(model_directory, filename)
        try:
            model = joblib.load(model_path)
            models[model_name] = model
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error loading model: {model_name} - {str(e)}"
            )


class Data(BaseModel):
    data: list


@app.post("/api/predict/test")
def predict(data: Data):
    # Check if the specified model exists
    if not models:
        raise HTTPException(status_code=500, detail="Models not found")
    # Reshape the input data
    X = np.array(data.data).reshape(1, -1)

    # Predict and store the results for each model
    predictions = {}

    for model_name, model in models.items():
        try:
            prediction = model.predict(X)
            prediction = np.expm1(prediction)
            predictions[model_name] = f"{prediction[0]:.6f} seconds"
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error predicting with model: {model_name} - {str(e)}",
            )

    # Format the prediction response as JSON
    response = {"predictions": predictions}
    return response
