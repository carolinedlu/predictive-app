import joblib
import os

# Specify the directory where the models are stored
model_directory = "./model"  # Change to "../model" if run on local machine


def load_models():
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
                raise RuntimeError(f"Error loading model: {model_name} - {str(e)}")

    if not models:
        raise RuntimeError("Models not found")

    return models


def load_model(model_name):
    model_path = os.path.join(model_directory, f"{model_name}.joblib")
    model = joblib.load(model_path)
    return model
