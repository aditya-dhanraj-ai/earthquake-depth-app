# 🌍 Earthquake Depth Predictor

A Machine Learning web application that predicts the depth of an earthquake based on its magnitude and geographical coordinates. Built with Streamlit and Scikit-Learn.

## 🚀 Features

- **Interactive UI:** User-friendly interface built with Streamlit.
- **Real-time Prediction:** Instantly predicts earthquake depth (in km) using a pre-trained Random Forest Regressor.
- **Machine Learning Integration:** Utilizes a saved `.pkl` model for fast inference without retraining.
- **Clean Architecture:** Separates the model training phase (done in Colab) from the deployment phase.

## 🛠️ Tech Stack

- **Python** (Core Language)
- **Streamlit** (Web Framework)
- **Scikit-Learn** (Machine Learning)
- **Pandas & NumPy** (Data Processing)
- **Joblib** (Model Serialization)

## 🧠 How it Works

1.  The model was trained on real-time USGS earthquake data (Magnitude > 4.5).
2.  Features used: `Magnitude`, `Latitude`, `Longitude`.
3.  Target Variable: `Depth (km)`.
4.  Algorithm: `RandomForestRegressor` (achieving an R2 Score of ~0.75).
5.  The trained model is saved as `depth_model.pkl` and loaded by the Streamlit app for live predictions.

## 💻 Live Demo

🔗 **[Click here to try the Live App](https://earthquake-depth-app.onrender.com/)** *(Note: Replace with your actual Render URL after deployment)*

## 👨‍💻 Developer

**Aditya Dhanraj**  
BCA Student | Aspiring ML Engineer  
*Preparing for MEXT Scholarship at Hokkaido University, Japan.*
