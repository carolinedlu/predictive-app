import joblib
import os

path = os.getcwd()


def load_models():
    # Load all models from the model directory
    models = {}
    for filename in os.listdir(path):
        if filename.endswith(".joblib"):
            model_name = os.path.splitext(filename)[0]
            model_path = os.path.join(path, filename)
            try:
                model = joblib.load(model_path)
                models[model_name] = model
            except Exception as e:
                raise RuntimeError(f"Error loading model: {model_name} - {str(e)}")

    if not models:
        raise RuntimeError("Models not found")

    return models


def load_model(model_path):
    model = joblib.load(model_path)
    return model
