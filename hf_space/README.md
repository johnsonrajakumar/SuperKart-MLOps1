---
title: SuperKart Sales Prediction
emoji: 🛒
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# SuperKart Sales Prediction

An ML-powered application for predicting product-store sales.

The application uses a trained machine learning pipeline hosted on the Hugging Face Model Hub and provides an interactive Streamlit interface for generating sales predictions.

## Model

The trained model is hosted separately on the Hugging Face Model Hub:

`johnsonrajkumar/superkart-sales-model`

## Application

The application accepts product and store attributes and generates a predicted product-store sales value.

## Technology Stack

- Python
- Streamlit
- Scikit-learn
- XGBoost
- Joblib
- Hugging Face Model Hub
- Docker
