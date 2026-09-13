
import os
import joblib
import pandas as pd
import streamlit as st

from huggingface_hub import hf_hub_download


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MODEL_REPO = os.getenv(
    "MODEL_REPO",
    "YOUR_USERNAME/superkart-sales-model"
)

MODEL_FILENAME = "superkart_sales_model.pkl"


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="SuperKart Sales Predictor",
    page_icon="📊",
    layout="centered"
)


# ---------------------------------------------------------
# Application title
# ---------------------------------------------------------

st.title("📊 SuperKart Sales Predictor")

st.write(
    """
    Predict product-store sales using the trained SuperKart
    machine learning model.
    """
)


# ---------------------------------------------------------
# Load model from Hugging Face
# ---------------------------------------------------------

@st.cache_resource
def load_model():

    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILENAME
    )

    model = joblib.load(model_path)

    return model


try:

    model = load_model()

    st.success(
        "Model loaded successfully from Hugging Face Model Hub."
    )

except Exception as e:

    st.error(
        f"Unable to load the model: {e}"
    )

    st.stop()


# ---------------------------------------------------------
# Input form
# ---------------------------------------------------------

st.subheader("Enter Product and Store Details")


with st.form("sales_prediction_form"):

    product_weight = st.number_input(
        "Product Weight",
        min_value=0.0,
        value=10.0,
        step=0.1
    )

    sugar_content = st.selectbox(
        "Product Sugar Content",
        [
            "Low Sugar",
            "Regular",
            "No Sugar"
        ]
    )

    allocated_area = st.number_input(
        "Product Allocated Area",
        min_value=0.0,
        max_value=1.0,
        value=0.05,
        step=0.001
    )

    product_type = st.selectbox(
        "Product Type",
        [
            "Baking Goods",
            "Breads",
            "Breakfast",
            "Canned",
            "Dairy",
            "Frozen Foods",
            "Fruits and Vegetables",
            "Hard Drinks",
            "Health and Hygiene",
            "Household",
            "Meat",
            "Others",
            "Seafood",
            "Snack Foods",
            "Soft Drinks",
            "Starchy Foods"
        ]
    )

    product_mrp = st.number_input(
        "Product MRP",
        min_value=0.0,
        value=150.0,
        step=1.0
    )

    store_id = st.selectbox(
        "Store ID",
        [
            "OUT001",
            "OUT002",
            "OUT003",
            "OUT004"
        ]
    )

    store_year = st.selectbox(
        "Store Establishment Year",
        [
            1987,
            1997,
            1999,
            2004
        ]
    )

    store_size = st.selectbox(
        "Store Size",
        [
            "Small",
            "Medium",
            "High"
        ]
    )

    city_type = st.selectbox(
        "Store Location City Type",
        [
            "Tier 1",
            "Tier 2",
            "Tier 3"
        ]
    )

    store_type = st.selectbox(
        "Store Type",
        [
            "Departmental Store",
            "Food Mart",
            "Supermarket Type1",
            "Supermarket Type2"
        ]
    )

    submitted = st.form_submit_button(
        "Predict Sales"
    )


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

if submitted:

    input_data = pd.DataFrame({
        "Product_Weight": [product_weight],
        "Product_Sugar_Content": [sugar_content],
        "Product_Allocated_Area": [allocated_area],
        "Product_Type": [product_type],
        "Product_MRP": [product_mrp],
        "Store_Id": [store_id],
        "Store_Establishment_Year": [store_year],
        "Store_Size": [store_size],
        "Store_Location_City_Type": [city_type],
        "Store_Type": [store_type]
    })

    st.subheader("Input Data")

    st.dataframe(
        input_data,
        use_container_width=True
    )

    try:

        prediction = model.predict(
            input_data
        )

        predicted_sales = float(
            prediction[0]
        )

        st.success(
            f"Predicted Sales: ₹{predicted_sales:,.2f}"
        )

    except Exception as e:

        st.error(
            f"Prediction failed: {e}"
        )
