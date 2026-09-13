
# SuperKart Sales Prediction Model

## Model Description

This model predicts `Product_Store_Sales_Total` using product and store
characteristics from the SuperKart dataset.

## Model

Random Forest

## Target

`Product_Store_Sales_Total`

## Evaluation Metrics

- MAE: 115.2639
- RMSE: 285.2186
- R²: 0.9287
- MAPE: 4.20%

## Features

Product_Weight, Product_Sugar_Content, Product_Allocated_Area, Product_Type, Product_MRP, Store_Id, Store_Establishment_Year, Store_Size, Store_Location_City_Type, Store_Type

## Usage

The saved Joblib artifact contains both preprocessing and the trained model.
It can therefore receive a dataframe containing the original model input
features and produce sales predictions.
