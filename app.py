import requests
import streamlit as st
import pandas as pd
from pydantic import BaseModel


class Data(BaseModel):
    data: list


def make_prediction(data_frame: pd.DataFrame):
    # Define the API endpoint for the FastAPI backend
    api_endpoint = "http://localhost:5001/api/predict/test"  # Modify the URL if needed

    # Convert dataFrame to dummy variables
    data = pd.get_dummies(data_frame)
    # return data

    # Prepare the request data
    request_data = {
        "data": data.values.tolist()
    }  # Convert the DataFrame to a nested list

    # Send a POST request to the API endpoint
    response = requests.post(api_endpoint, json=request_data)

    # Parse the response as JSON
    predictions = response.json()["predictions"]

    return predictions


def main():
    st.title("Predictive App")

    major_diameter = st.number_input("Major Diameter", value=0.0)
    minor_diameter = st.number_input("Minor Diameter", value=0.0)
    thread_pitch = st.number_input("Thread Pitch", value=0.0)
    included_angle = st.number_input("Included Angle", value=0.0)
    top_angle = st.number_input("Top Angle", value=0.0)
    hole_diameter = st.number_input("Hole Diameter", value=0.0)
    serration_threads = st.radio(
        "Serration Threads", options=[False, True], index=False
    )
    if serration_threads:
        serration_threads = 1
    else:
        serration_threads = 0
    length_B = st.number_input("Length B", value=0.0)
    length_C = st.number_input("Length C", value=0.0)
    embedment_depth = st.number_input("Embedment Depth", value=0.0)
    tip_taper = st.number_input("Tip Taper", value=0.0)
    shank_diameter = st.number_input("Shank Diameter", value=0.0)
    torque_tool = st.selectbox("Torque Tool", options=["Low", "Medium", "High"])
    if torque_tool == "Low":
        torque_tool = 225
    elif torque_tool == "Medium":
        torque_tool = 300
    else:
        torque_tool = 500
    concrete_tx = st.radio("Texas Concrete", options=[False, True], index=False)
    if concrete_tx:
        concrete_tx = 1
    else:
        concrete_tx = 0
    concrete_wc = st.radio("West Chicago Concrete", options=[False, True], index=False)
    if concrete_wc:
        concrete_wc = 1
    else:
        concrete_wc = 0

    input = {
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
        "concrete_tx": concrete_tx,
        "concrete_wc": concrete_wc,
    }

    # Input conversion to DataFrame
    data_frame = pd.DataFrame([input])

    if st.button("Predict"):
        predictions = make_prediction(data_frame)
        # st.write(predictions)
        st.write("Predictions:")
        for model_name, prediction in predictions.items():
            st.write(f"{model_name}: {prediction}")


if __name__ == "__main__":
    main()
