import streamlit as st
import joblib

# Load model
model = joblib.load("final_model.pkl")

st.title("🔥 Forest Fire Risk Detection")

st.write("Enter the values:")

lat = st.number_input("Latitude")
lon = st.number_input("Longitude")
temp_mean = st.number_input("Mean Temperature")
humidity_min = st.number_input("Minimum Humidity")
wind_speed_max = st.number_input("Maximum Wind Speed")

input_data = [[lat, lon, temp_mean, humidity_min, wind_speed_max]]

if st.button("Predict"):
    prediction = model.predict(input_data)
    st.success(f"Prediction: {'🔥 Fire Occurred' if prediction[0]==1 else '✅ No Fire'}")
