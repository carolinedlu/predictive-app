import warnings
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning

warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
warnings.simplefilter("ignore", category=NumbaPendingDeprecationWarning)

import os
import numpy as np
import streamlit as st
import pandas as pd
from pydantic import BaseModel
import shap
import joblib
from streamlit_shap import st_shap
from model_loader import load_models, load_model

models = load_models()


class Data(BaseModel):
    data: list


def preprocess_data(data_frame):
    # Convert DataFrame to dummy variables
    data = pd.get_dummies(data_frame)
    return data


def make_prediction(data_frame):
    data = preprocess_data(data_frame)

    X = np.array(data).reshape(1, -1)

    predictions = {}

    for model_name, model in models.items():
        try:
            prediction = model.predict(X)
            prediction = np.expm1(prediction)
            predictions[model_name] = f"{prediction[0]:.6f} seconds"
            return predictions
        except Exception as e:
            st.error(f"Error predicting with model: {model_name} - {str(e)}")

    return None


def main():
    st.title("Anchor Installation Time Predict")
    st.header("Input Parameters")
    major_diameter = st.number_input(
        "Major Diameter (in.)", min_value=0.0, value=0.0, format="%.4f", step=1e-4
    )
    minor_diameter = st.number_input(
        "Minor Diameter (in.)", min_value=0.0, value=0.0, format="%.4f", step=1e-4
    )
    thread_pitch = st.number_input(
        "Thread Pitch (in.)", min_value=0.0, value=0.0, format="%.4f", step=1e-4
    )
    included_angle = st.number_input(
        "Included Angle", min_value=0.0, value=0.0, format="%.4f", step=1e-4
    )
    top_angle = st.number_input(
        "Top Angle", min_value=0.0, value=0.0, format="%.4f", step=1e-4
    )
    hole_diameter = st.number_input(
        "Hole Diameter (in.)", min_value=0.0, value=0.0, format="%.4f", step=1e-4
    )
    serration_threads = st.radio(
        "Have Serration Threads", options=[False, True], index=False
    )
    serration_threads = int(serration_threads)  # Convert to int
    length_B = st.number_input(
        "Total length (in.)", min_value=0.0, value=0.0, format="%.4f", step=1e-4
    )
    length_C = st.number_input(
        "Thread length (in.)", min_value=0.0, value=0.0, format="%.4f", step=1e-4
    )
    embedment_depth = st.number_input(
        "Embedment Depth (in.)", min_value=0.0, value=0.0, format="%.4f", step=1e-4
    )
    tip_taper = st.number_input(
        "Tip Taper", min_value=0.0, value=0.0, format="%.4f", step=1e-4
    )
    shank_diameter = st.number_input(
        "Shank Diameter (in.)", min_value=0.0, value=0.0, format="%.4f", step=1e-4
    )
    torque_tool = st.selectbox("Torque Tool", options=["Low", "Medium", "High"])
    torque_values = {"Low": 225, "Medium": 300, "High": 500}
    torque_tool = torque_values.get(torque_tool, 225)  # Default to Low torque
    # concrete = st.radio("Concrete", options=["Texas", "West Chicago"])
    # concrete_values = {"Texas": (1, 0), "West Chicago": (0, 1)}
    # concrete_tx, concrete_wc = concrete_values.get(concrete, (1, 0))  # Default to Texas

    input_data = {
        "major_diameter": major_diameter,
        "minor_diameter": minor_diameter,
        "thread_pitch": thread_pitch,
        "included_angle": included_angle,
        "top_angle": top_angle,
        "hole_diameter": hole_diameter,
        "serration_threads": serration_threads,
        "length_B": length_B,
        "length_C": length_C,
        "embedment_depth": embedment_depth,
        "tip_taper": tip_taper,
        "shank_diameter": shank_diameter,
        "torque_tool": torque_tool,
        # "concrete_tx": concrete_tx,
        # "concrete_wc": concrete_wc,
    }

    # Check if any input value is zero or empty
    if (
        major_diameter == 0.0
        or minor_diameter == 0.0
        or thread_pitch == 0.0
        or included_angle == 0.0
        or top_angle == 0.0
        or hole_diameter == 0.0
        or length_B == 0.0
        or length_C == 0.0
        or embedment_depth == 0.0
        # or tip_taper == 0.0
        or shank_diameter == 0.0
    ):
        st.error("Error: Please fill in all the input fields.")
        return

    data_frame = pd.DataFrame([input_data])
    # st.write(data_frame)

    show_results = st.checkbox("Show Models' results")

    if show_results:
        predictions = make_prediction(data_frame)

        if predictions is not None:
            st.header("Predictions:")

            # Create a table to display the predictions
            table_data = []
            for model_name, prediction in predictions.items():
                table_data.append([model_name, prediction])

            table_df = pd.DataFrame(
                table_data, columns=["Model", "Installing time prediction"]
            )

            # Display the table
            st.table(table_df)

            # Create a radio button or selectbox for users to select a model
            model_options = list(predictions.keys())
            selected_model = st.radio(
                "Select a model to generate SHAP plot:", model_options
            )

            if st.button(f"Generate SHAP plot for {selected_model}"):
                # Load the selected model from the selected_model
                model = load_model(selected_model)
                # Generate the SHAP plot for the selected model
                explainer = shap.TreeExplainer(model)

                data = preprocess_data(data_frame)

                shap_values = explainer(data)
                # st.write(shap_values)

                # Show the SHAP plot
                st_shap(
                    shap.plots.waterfall(shap_values[0], max_display=20), height=600
                )


if __name__ == "__main__":
    main()
