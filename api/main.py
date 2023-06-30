from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import sys

sys.path.append("../app")  # Add the "app" directory to the Python path
from model_loader import load_models

app = FastAPI()

models = load_models()


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
