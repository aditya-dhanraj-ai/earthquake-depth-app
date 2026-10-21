import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Earthquake Depth Predictor", page_icon="🌍", layout="centered")

@st.cache_resource
def load_model():
    model = joblib.load('depth_model.pkl')
    return model

model = load_model()

st.markdown("<h1 style='text-align: center; color: #ff4b4b;'>🌍 Earthquake Depth Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #888;'>Enter earthquake parameters to predict its depth</p>", unsafe_allow_html=True)
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    magnitude = st.number_input("Magnitude", min_value=1.0, max_value=10.0, value=5.0, step=0.1)
with col2:
    lat = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=35.0, step=0.1)
with col3:
    lon = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=140.0, step=0.1)

st.markdown("---")

if st.button("🔮 Predict Depth", use_container_width=True):
    features = np.array([[magnitude, lat, lon]])
    
    prediction = model.predict(features)
    
    st.success(f"✅ Predicted Depth: **{prediction[0]:.2f} km**")
    st.info("Note: This is an ML-based prediction using Random Forest Regressor.")
