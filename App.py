import streamlit as st
import joblib
import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim

st.set_page_config(page_title="Earthquake Depth Predictor", page_icon="🌍", layout="wide")

st.markdown("""
<style>
    /* Dark Theme Background */
    .stApp {
        background: linear-gradient(135deg, #0e1117 0%, #1a1f2e 100%);
    }
    /* Title Styling */
    h1 {
        color: #ff4b4b;
        text-align: center;
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    /* Subtitle Styling */
    .subtitle {
        text-align: center;
        color: #a0aab5;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    /* Input Box Styling */
    div[data-testid="stNumberInput"] > div {
        background-color: #1e2530;
        border-radius: 10px;
        border: 1px solid #2a3441;
    }
    /* Button Styling */
    div.stButton > button {
        background: linear-gradient(90deg, #ff4b4b 0%, #d32f2f 100%);
        color: white;
        border-radius: 8px;
        font-weight: bold;
        font-size: 1.1rem;
        padding: 0.6rem 2rem;
        border: none;
        width: 100%;
        transition: 0.3s;
    }
    div.stButton > button:hover {
        background: linear-gradient(90deg, #d32f2f 0%, #ff4b4b 100%);
        transform: scale(1.02);
    }
    /* Result Box Styling */
    .result-box {
        background: linear-gradient(135deg, #1e2530 0%, #2a3441 100%);
        padding: 25px;
        border-radius: 12px;
        border-left: 6px solid #00d4ff;
        box-shadow: 0px 4px 15px rgba(0,212,255,0.2);
        text-align: center;
        margin-top: 20px;
    }
    .result-text {
        font-size: 2rem;
        font-weight: bold;
        color: #00d4ff;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load('depth_model.pkl')

model = load_model()

geolocator = Nominatim(user_agent="earthquake_app")

st.markdown("<h1>🌍 Earthquake Depth Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>AI-powered seismic parameter estimation using Random Forest</p>", unsafe_allow_html=True)
st.markdown("---")

col_input, col_output = st.columns([1, 1.5], gap="large")

with col_input:
    st.markdown("### 📥 Input Parameters")
    magnitude = st.number_input("Magnitude", min_value=1.0, max_value=10.0, value=5.0, step=0.1)
    lat = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=35.0, step=0.1)
    lon = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=140.0, step=0.1)
    
    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔮 Predict Depth")

with col_output:
    st.markdown("### 📊 Prediction & Analysis")
    
    if predict_btn:
        with st.spinner("AI Model is calculating..."):
            features = np.array([[magnitude, lat, lon]])
            prediction = model.predict(features)[0]
            
            st.markdown(f"""
                <div class="result-box">
                    <p style="color: #a0aab5; margin-bottom: 5px;">Predicted Depth</p>
                    <p class="result-text">{prediction:.2f} km</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            try:
                location = geolocator.reverse(f"{lat}, {lon}")
                location_name = location.address if location else "Unknown Ocean/Region"
                st.info(f"📍 **Location:** {location_name}")
            except:
                st.warning("⚠️ Location name fetch nahi ho paya (Network issue).")
            
            
            map_data = pd.DataFrame({'lat': [lat], 'lon': [lon]})
            st.map(map_data, zoom=6, use_container_width=True)
            
    else:
        st.info("👈 Please enter parameters and click 'Predict Depth' to see the magic!")

st.markdown("---")
st.markdown("<p style='text-align: center; color: #555;'>Built with Streamlit & Scikit-Learn</p>", unsafe_allow_html=True)
