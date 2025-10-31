import streamlit as st
import requests
import pandas as pd
import json

# --- Page Configuration ---
st.set_page_config(
    page_title="California House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# --- Configuration ---
API_URL = "http://127.0.0.1:8000/predict"

# --- Main Application ---
def run_app():
    
    # 1. Header
    st.title("🏠 California House Price Predictor")
    st.markdown("Enter the details for a block group in California to get an estimated median house value.")
    st.info("This model is trained on the `sklearn.datasets.fetch_california_housing` dataset.")
    
    # 2. User Inputs
    with st.form(key="prediction_form"):
        st.subheader("Enter Block Group Features:")
        
        # We need 8 inputs now. We can use columns to organize them.
        col1, col2 = st.columns(2)
        
        with col1:
            MedInc = st.number_input("Median Income ($10,000s)", min_value=0.0, max_value=20.0, value=3.87, step=0.1, key="MedInc")
            AveRooms = st.number_input("Average Rooms", min_value=1.0, max_value=20.0, value=5.8, step=0.1, key="AveRooms")
            Population = st.number_input("Population", min_value=1, max_value=50000, value=1425, step=100, key="Population")
            Latitude = st.number_input("Latitude", min_value=32.0, max_value=42.0, value=37.88, step=0.01, key="Latitude")

        with col2:
            HouseAge = st.number_input("Median House Age", min_value=1, max_value=60, value=21, step=1, key="HouseAge")
            AveBedrms = st.number_input("Average Bedrooms", min_value=0.5, max_value=10.0, value=1.04, step=0.1, key="AveBedrms")
            AveOccup = st.number_input("Average Occupancy", min_value=1.0, max_value=20.0, value=2.55, step=0.1, key="AveOccup")
            Longitude = st.number_input("Longitude", min_value=-125.0, max_value=-114.0, value=-122.23, step=0.01, key="Longitude")

        # Submit button for the form
        submit_button = st.form_submit_button(label="Predict Value")

    # 3. Form Submission Logic
    if submit_button:
        # Create the payload to send to the API
        payload = {
            "MedInc": MedInc,
            "HouseAge": HouseAge,
            "AveRooms": AveRooms,
            "AveBedrms": AveBedrms,
            "Population": Population,
            "AveOccup": AveOccup,
            "Latitude": Latitude,
            "Longitude": Longitude
        }
        
        # Display a spinner while waiting for the API response
        with st.spinner("Calling API... 🚀"):
            try:
                # Make the POST request
                response = requests.post(API_URL, json=payload, timeout=5)
                
                # Check for successful response
                if response.status_code == 200:
                    result = response.json()
                    value_100k = result.get("predicted_value_100k")
                    
                    # Convert the value (which is in $100,000s) to a full dollar amount
                    full_value = value_100k * 100000
                    
                    st.success(f"**Predicted Median House Value:** \n## ${full_value:,.2f}")
                
                else:
                    # Show API error
                    st.error(f"Error from API: {response.status_code} - {response.text}")
            
            except requests.exceptions.ConnectionError:
                st.error("Connection Error: Could not connect to the API. "
                         "Please ensure the FastAPI server is running at 'http://127.0.0.1:8000'.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    run_app()

