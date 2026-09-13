
import os
import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download


# ============================================================
# Configuration
# ============================================================

MODEL_REPO = os.getenv(
    "MODEL_REPO",
    "johnsonrajakumar/superkart-sales-model"
)

MODEL_FILENAME = "superkart_sales_model.pkl"


# ============================================================
# Streamlit Page Configuration
# ============================================================

st.set_page_config(
    page_title="SuperKart Sales Prediction",
    page_icon="🛒",
    layout="centered"
)


# ============================================================
# Load Model from Hugging Face Model Hub
# ============================================================

@st.cache_resource
def load_model():

    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILENAME,
        repo_type="model"
    )

    model = joblib.load(model_path)

    return model


# Load trained model
model = load_model()


# ============================================================
# Application Header
# ============================================================

st.title("🛒 SuperKart Sales Prediction")

st.write(
    "Enter the product and store details below to predict "
    "the total product-store sales."
)


# ============================================================
# Input Form
# ============================================================

with st.form("sales_prediction_form"):

    st.subheader("Product Information")

    product_weight = st.number_input(
        "Product Weight",
        min_value=0.0,
        value=12.5,
        step=0.1
    )

    product_sugar_content = st.selectbox(
        "Product Sugar Content",
        [
            "Low Sugar",
            "Regular",
            "No Sugar"
        ]
    )

    product_allocated_area = st.number_input(
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
            "Health and Hygiene",
            "Household",
            "Meat",
            "Others",
            "Seafood",
            "Snack Foods",
            "Soft Drinks",
            "Starchy Foods",
            "Hard Drinks"
        ]
    )

    product_mrp = st.number_input(
        "Product MRP",
        min_value=0.0,
        value=150.0,
        step=1.0
    )

    st.subheader("Store Information")

    store_id = st.selectbox(
        "Store ID",
        [
            "OUT001",
            "OUT002",
            "OUT003",
            "OUT004"
        ]
    )

    store_establishment_year = st.selectbox(
        "Store Establishment Year",
        [
            1987,
            1998,
            1999,
            2009
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

    store_location_city_type = st.selectbox(
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
            "Supermarket Type1",
            "Supermarket Type2",
            "Departmental Store",
            "Food Mart"
        ]
    )

    submitted = st.form_submit_button(
        "Predict Sales"
    )


# ============================================================
# Prediction
# ============================================================

if submitted:

    # Create input DataFrame
    input_data = pd.DataFrame([{
        "Product_Weight": product_weight,
        "Product_Sugar_Content": product_sugar_content,
        "Product_Allocated_Area": product_allocated_area,
        "Product_Type": product_type,
        "Product_MRP": product_mrp,
        "Store_Id": store_id,
        "Store_Establishment_Year": store_establishment_year,
        "Store_Size": store_size,
        "Store_Location_City_Type": store_location_city_type,
        "Store_Type": store_type
    }])

    # Generate prediction
    prediction = model.predict(input_data)[0]

    # Display prediction
    st.success(
        f"Predicted Sales: ₹{prediction:,.2f}"
    )

    # Display submitted values
    st.subheader("Input Details")

    st.dataframe(
        input_data,
        use_container_width=True
    )
