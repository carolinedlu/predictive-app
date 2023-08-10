import os

import warnings
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning

warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
warnings.simplefilter("ignore", category=NumbaPendingDeprecationWarning)

import streamlit as st
import pandas as pd
import shap
from streamlit_shap import st_shap

from predict import make_prediction, preprocess_data

# sys.path.append(os.path.abspath(os.path.join("..", "model")))

from main.model.model_loader import load_model

statistics_path = os.path.join("statistics.json")


def generate_shap_plot(model_name, data):
    model_path = os.path.join("..", "model", f"{model_name}.joblib")
    # Load the selected model from the selected_model
    model = load_model(model_path)
    # Generate the SHAP plot for the selected model
    explainer = shap.TreeExplainer(model)

    standardize_data = preprocess_data(data, statistics_path)
    shap_values = explainer(standardize_data)

    # Show the SHAP plot
    st_shap(shap.plots.waterfall(shap_values[0], max_display=20), height=600)


def main():
    st.title("Anchor Installation Time Predict")
    st.header("Input Parameters")
    # major_diameter = st.number_input(
    #     "Major Diameter (in.)", min_value=0.0, value=0.0, format="%.4f", step=1e-4
    # )
    # minor_diameter = st.number_input(
    #     "Minor Diameter (in.)", min_value=0.0, value=0.0, format="%.4f", step=1e-4
    # )
    undercut = st.number_input(
        "Undercut (in.)", min_value=0.0, value=0.0, format="%.4f", step=1e-4
    )
    gap = st.number_input(
        "Gap (in.)", min_value=0.0, value=0.0, format="%.4f", step=1e-4
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
    # hole_diameter = st.number_input(
    #     "Hole Diameter (in.)", min_value=0.0, value=0.0, format="%.4f", step=1e-4
    # )
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
    # shank_diameter = st.number_input(
    #     "Shank Diameter (in.)", min_value=0.0, value=0.0, format="%.4f", step=1e-4
    # )
    torque_tool = st.selectbox("Torque Tool", options=["Low", "Medium", "High"])
    torque_values = {"Low": 225, "Medium": 300, "High": 500}
    torque_tool = torque_values.get(torque_tool, 225)  # Default to Low torque
    concrete = st.number_input("Concrete Strength (psi)", min_value=0.0, value=0.0)

    input_data = {
        # "Major Diameter": major_diameter,
        # "Minor Diameter": minor_diameter,
        "Pitch": thread_pitch,
        "Included Angle": included_angle,
        "Top Angle": top_angle,
        # "Hole Diameter": hole_diameter,
        "Serrations": serration_threads,
        "Length (C)": length_C,
        "Length (B)": length_B,
        "Embedment depth": embedment_depth,
        "Tip Taper": tip_taper,
        # "Shank Diameter": shank_diameter,
        "Torque Tool": torque_tool,
        "Concrete": concrete,
        "Undercut": undercut,
        "Gap": gap,
    }

    # Hard code the input data for testing
    # input_data = {
    #     "Pitch": 0.52,
    #     "Included Angle": 55,
    #     "Top Angle": 27,
    #     "Serrations": 0,
    #     "Length (C)": 5.5,
    #     "Length (B)": 6.5,
    #     "Embedment depth": 5.5,
    #     "Tip Taper": 0.0,
    #     "Torque Tool": 225,
    #     "Concrete": 5500,
    #     "Undercut": 0.074,
    #     "Gap": 0.074,
    # }

    # Check if any input value is zero or empty
    if (
        # major_diameter == 0.0
        # or minor_diameter == 0.0
        undercut == 0.0
        or gap == 0.0
        or thread_pitch == 0.0
        or included_angle == 0.0
        or top_angle == 0.0
        # or hole_diameter == 0.0
        or length_B == 0.0
        or length_C == 0.0
        or embedment_depth == 0.0
        # or tip_taper == 0.0
        # or shank_diameter == 0.0
    ):
        st.error("Error: Please fill in all the input fields.")
        return

    data_frame = pd.DataFrame([input_data])
    # st.write(data_frame)

    show_results = st.button("Predict")

    # Initialize session state
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = None

    if show_results or st.session_state.selected_model is not None:
        st.session_state.selected_model = True
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

            # Create a radio button for users to select a model
            model_options = list(predictions.keys())
            selected_model = st.radio(
                "Select a model to generate SHAP plot:", model_options
            )

            generate_shap_button_clicked = st.button(
                f"Generate SHAP plot for {selected_model}"
            )

            if generate_shap_button_clicked:
                if selected_model is not None:
                    generate_shap_plot(selected_model, data_frame)


if __name__ == "__main__":
    main()
