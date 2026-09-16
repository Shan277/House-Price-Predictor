# Ames Housing – Sale Price Predictor 🏠

Predict a home's sale price from the Ames Housing dataset using engineered features and a regularized linear pipeline, with a Streamlit app for interactive predictions.

## 🛠️ Tools & Tech Stack

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Pandas](https://img.shields.io/badge/Pandas-DataFrame-150458)
![NumPy](https://img.shields.io/badge/NumPy-Array-013243)
![scikit--learn](https://img.shields.io/badge/scikit--learn-Pipeline%20%7C%20Ridge%20%7C%20GridSearchCV-F7931E)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Plotting-11557C)
![Seaborn](https://img.shields.io/badge/Seaborn-EDA-4C72B0)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B)
![Joblib](https://img.shields.io/badge/Joblib-Model%20Serialization-8A2BE2)

## 📌 Overview

This project analyzes the Ames Housing dataset to identify what drives home sale prices, and builds a regression pipeline to predict them. It includes:

- Exploratory Data Analysis (EDA) covering distributions, skewness, and correlation with `SalePrice`
- Feature engineering to combine related raw features into more informative ones
- A nested preprocessing pipeline handling numeric, log-skewed, ordinal, and nominal features differently
- Cross-validation and hyperparameter tuning on a log-transformed target
- A Streamlit app that predicts sale price from a compact set of the most important inputs

## 📊 Dataset

The dataset (`train.csv`) is the classic Ames Housing dataset, with ~80 features describing each home (lot size, quality ratings, basement/garage details, neighborhood, sale conditions, etc.) and a `SalePrice` target.

## 🔍 Exploratory Data Analysis

Key steps performed in the notebook:
- Checked shape, data types, summary statistics, and confirmed no duplicate rows
- Quantified missing values per feature to plan imputation strategy
- Found `SalePrice` is right-skewed and applied a log transform (`log1p`) to normalize it
- Identified and visualized other heavily skewed numerical features (skew > 0.75)
- Built a correlation heatmap and ranked features by correlation with `SalePrice`
- Scatter-plotted key features (`GrLivArea`, `TotalBsmtSF`, `GarageArea`, `1stFlrSF`) against `SalePrice` and flagged outliers

## 🏗️ Feature Engineering

- `TotalSF` = `TotalBsmtSF` + `1stFlrSF` + `2ndFlrSF`
- `TotalFullBath` = `FullBath` + `BsmtFullBath`
- `TotalHalfBath` = `BsmtHalfBath` + `HalfBath`
- `HouseAge` = `YrSold` − `YearBuilt`
- Dropped the original components of the above (plus `Id`, `GarageArea`, `TotRmsAbvGrd`, `GarageYrBlt`) once the combined features were created
- Removed outliers (`GrLivArea`, `TotalBsmtSF`, `1stFlrSF` ≥ 4000 sqft) and the single row with a missing `Electrical` value

## ⚙️ Preprocessing & Modeling Pipeline

Built as a nested `scikit-learn` `Pipeline` with a `ColumnTransformer` combining four branches:

| Branch | Features | Steps |
|---|---|---|
| Ordinary numerical | Non-skewed numeric features | Median impute → `StandardScaler` |
| Log-transformed numerical | Skewed numeric features (skew > 0.75) | Median impute → `log1p` → `StandardScaler` |
| Ordinal | Quality/condition-style columns (e.g. `ExterQual`, `BsmtQual`, `KitchenQual`) | Constant impute (`"None"`) → `OrdinalEncoder` with explicit category ordering |
| Nominal | Categorical columns (e.g. `Neighborhood`, `Exterior1st`, `SaleCondition`) | Constant impute (`"None"`) → `OneHotEncoder(handle_unknown="ignore")` |

The target (`SalePrice`) is log-transformed (`log1p`) before training and predictions are converted back with `expm1`.

**Model:** `Ridge` regression, tuned via `GridSearchCV` over `alpha` with 5-fold cross-validation (scored on R²), then evaluated on the held-out test set with **R², RMSE, and MAE**. The fitted pipeline is serialized with `joblib` (`model.pkl`).

## 🖥️ Streamlit App

`app.py` loads the trained pipeline and starts from a full default row covering every feature the model expects. It only exposes the ~12 features most correlated with `SalePrice` (overall quality, living area, total square footage, garage cars, bathrooms, bedrooms, condition, year remodeled, house age, fireplaces, finished basement area, masonry veneer area) as editable inputs — everything else keeps its default. Predictions are converted back from log scale before display.

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/ames-housing-price-predictor.git
cd ames-housing-price-predictor
```

### 2. Install dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit joblib
```

### 3. Train the model (optional — a pretrained `model.pkl` can be used instead)
Run through `HousePricePredictor.ipynb` to reproduce the EDA, feature engineering, and model training.

### 4. Run the app
```bash
streamlit run app.py
```

## 📂 Project Structure

```
├── HousePricePredictor.ipynb   # EDA, feature engineering, pipeline, tuning & evaluation
├── app.py                      # Streamlit app for predictions
├── model.pkl                   # Serialized trained pipeline
└── README.md
```

## 📈 Possible Improvements

- Compare Ridge/Lasso against gradient-boosted models (XGBoost, LightGBM) or a stacked ensemble
- Expand hyperparameter search beyond `alpha` (e.g. polynomial features, feature selection)
- Add residual and prediction-error plots for deeper model diagnostics
- Deploy the app (Streamlit Community Cloud / Docker)

## 📝 License

This project is open-sourced for educational purposes.
