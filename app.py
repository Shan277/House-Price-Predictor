import streamlit as st
import numpy as np
import pandas as pd
import pickle
from pandas.api.types import CategoricalDtype

# -------------------------------------------------------------------
# STEP 1: Load model, scaler and column names
# -------------------------------------------------------------------
with open("house_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("column_names.pkl", "rb") as f:
    column_names = pickle.load(f)

# -------------------------------------------------------------------
# STEP 2: Page title
# -------------------------------------------------------------------
st.title("🏠 House Price Predictor")

# -------------------------------------------------------------------
# STEP 3: Collect inputs from the user
# -------------------------------------------------------------------
st.header("Enter House Details")

overall_qual  = st.slider("Overall Quality (1 = worst, 10 = best)", 1, 10, 5)
overall_cond  = st.slider("Overall Condition (1 = worst, 10 = best)", 1, 10, 5)
year_built    = st.number_input("Year Built", min_value=1872, max_value=2010, value=2000)
gr_liv_area   = st.number_input("Living Area (sq ft)", min_value=500, max_value=6000, value=1500)
total_bsmt_sf = st.number_input("Basement Area (sq ft)", min_value=0, max_value=3000, value=800)
garage_cars   = st.number_input("Garage Size (no. of cars)", min_value=0, max_value=4, value=2)
full_bath     = st.number_input("Number of Full Bathrooms", min_value=0, max_value=4, value=2)
bedroom_abvgr = st.number_input("Number of Bedrooms", min_value=0, max_value=10, value=3)
kitchen_qual  = st.selectbox("Kitchen Quality", ["Po", "Fa", "TA", "Gd", "Ex"], index=2,
                              help="Po=Poor, Fa=Fair, TA=Average, Gd=Good, Ex=Excellent")

# -------------------------------------------------------------------
# STEP 4: When button is clicked, preprocess and predict
# -------------------------------------------------------------------
if st.button("Predict Price"):

    # 4a. Build one row of raw data
    row = {
        "MSSubClass": "60",        "MSZoning": "RL",
        "LotFrontage": 69.0,       "LotArea": 8450,
        "Street": "Pave",          "Alley": "NA",
        "LotShape": "Reg",         "LandContour": "Lvl",
        "Utilities": "AllPub",     "LotConfig": "Inside",
        "LandSlope": "Gtl",        "Neighborhood": "NAmes",
        "Condition1": "Norm",      "Condition2": "Norm",
        "BldgType": "1Fam",        "HouseStyle": "2Story",

        "OverallQual":  overall_qual,
        "OverallCond":  overall_cond,
        "YearBuilt":    str(year_built),
        "YearRemodAdd": str(year_built),

        "RoofStyle": "Gable",      "RoofMatl": "CompShg",
        "Exterior1st": "VinylSd",  "Exterior2nd": "VinylSd",
        "MasVnrType": "None",      "MasVnrArea": 0.0,
        "ExterQual": "TA",         "ExterCond": "TA",
        "Foundation": "PConc",

        "BsmtQual": "Gd",          "BsmtCond": "TA",
        "BsmtExposure": "No",      "BsmtFinType1": "GLQ",
        "BsmtFinSF1": 0.0,         "BsmtFinType2": "Unf",
        "BsmtFinSF2": 0.0,
        "BsmtUnfSF":   total_bsmt_sf,
        "TotalBsmtSF": total_bsmt_sf,

        "Heating": "GasA",         "HeatingQC": "Ex",
        "CentralAir": "Y",         "Electrical": "SBrkr",
        "1stFlrSF": gr_liv_area // 2,
        "2ndFlrSF": gr_liv_area // 2,
        "LowQualFinSF": 0,
        "GrLivArea": gr_liv_area,

        "BsmtFullBath": 0.0,       "BsmtHalfBath": 0.0,
        "FullBath": full_bath,
        "HalfBath": 0,
        "BedroomAbvGr": bedroom_abvgr,
        "KitchenAbvGr": 1,
        "KitchenQual":  kitchen_qual,
        "TotRmsAbvGrd": bedroom_abvgr + 2,

        "Functional": "Typ",       "Fireplaces": 0,
        "FireplaceQu": "NA",       "GarageType": "Attchd",
        "GarageYrBlt": str(year_built),
        "GarageFinish": "RFn",
        "GarageCars": garage_cars,
        "GarageArea": garage_cars * 240,

        "GarageQual": "TA",        "GarageCond": "TA",
        "PavedDrive": "Y",
        "WoodDeckSF": 0,           "OpenPorchSF": 0,
        "EnclosedPorch": 0,        "3SsnPorch": 0,
        "ScreenPorch": 0,          "PoolArea": 0,
        "PoolQC": "NA",            "Fence": "NA",
        "MiscFeature": "NA",       "MiscVal": 0,
        "MoSold": "Jun",           "YrSold": "2008",
        "SaleType": "WD",          "SaleCondition": "Normal",
    }

    df = pd.DataFrame([row])

    # 4b. Ordinal encoding — same as notebook
    ordinal_mappings = {
        "ExterQual":    ["Po", "Fa", "TA", "Gd", "Ex"],
        "ExterCond":    ["Po", "Fa", "TA", "Gd", "Ex"],
        "HeatingQC":    ["Po", "Fa", "TA", "Gd", "Ex"],
        "KitchenQual":  ["Po", "Fa", "TA", "Gd", "Ex"],
        "BsmtQual":     ["NA", "Po", "Fa", "TA", "Gd", "Ex"],
        "BsmtCond":     ["NA", "Po", "Fa", "TA", "Gd", "Ex"],
        "FireplaceQu":  ["Na", "Po", "Fa", "TA", "Gd", "Ex"],
        "GarageQual":   ["Na", "Po", "Fa", "TA", "Gd", "Ex"],
        "GarageCond":   ["Na", "Po", "Fa", "TA", "Gd", "Ex"],
        "PoolQC":       ["NA", "Fa", "TA", "Gd", "Ex"],
        "BsmtExposure": ["NA", "No", "Mn", "Av", "Gd"],
        "BsmtFinType1": ["NA", "Unf", "LwO", "Rec", "BLQ", "ALQ", "GLQ"],
        "BsmtFinType2": ["NA", "Unf", "LwO", "Rec", "BLQ", "ALQ", "GLQ"],
        "GarageFinish": ["NA", "Unf", "RFn", "Fin"],
        "PavedDrive":   ["N", "P", "Y"],
        "Utilities":    ["ELO", "NoSeWa", "NoSewr", "AllPub"],
        "Functional":   ["Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"],
    }

    for column, order in ordinal_mappings.items():
        df[column] = df[column].astype(
            CategoricalDtype(categories=order, ordered=True)
        ).cat.codes

    # 4c. One-hot encoding — same as notebook
    text_columns = df.select_dtypes(include="object").columns.tolist()
    df = pd.get_dummies(df, columns=text_columns, drop_first=True)

    # 4d. ✅ THE FIX — reindex using saved column names
    #     Every column lands in the EXACT same position as during training
    #     Missing columns get 0, extra columns get dropped
    df = df.reindex(columns=column_names, fill_value=0)

    # 4e. ✅ THE FIX — scale using the saved scaler (same as training)
    df_scaled = scaler.transform(df)

    # 4f. Predict
    predicted_price = model.predict(df_scaled)[0]

    # -------------------------------------------------------------------
    # STEP 5: Show the result
    # -------------------------------------------------------------------
    st.success(f"Estimated Sale Price:  ${predicted_price:,.0f}")