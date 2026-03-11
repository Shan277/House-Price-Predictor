# 🏠 House Price Predictor

![Python](https://img.shields.io/badge/PYTHON-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/STREAMLIT-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/SCIKIT--LEARN-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/PANDAS-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NUMPY-013243?style=for-the-badge&logo=numpy&logoColor=white)

A machine learning project that predicts residential house sale prices using the classic **Ames Housing Dataset**. The core focus is a carefully engineered ML pipeline — covering missing value imputation, feature encoding, scaling, multi-model benchmarking, and hyperparameter tuning — culminating in a tuned **Gradient Boosting Regressor** exported for deployment.

---

## 📁 Project Structure

```
house-price-predictor/
│
├── House-Prediction.ipynb   # Full ML pipeline (EDA → training → export)
├── app.py                   # Streamlit web application
├── train.csv                # Training data (Ames Housing Dataset)
├── test.csv                 # Test data
│
├── house_model.pkl          # Serialized trained model
├── scaler.pkl               # Serialized StandardScaler
└── column_names.pkl         # Serialized training column order
```

---

## ⚙️ Tech Stack

| Layer | Library / Tool |
|---|---|
| Data Processing | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn` |
| Encoding | `pandas.CategoricalDtype` |
| Scaling | `sklearn.preprocessing.StandardScaler` |
| Modeling | `scikit-learn`, `xgboost` |
| Hyperparameter Tuning | `RandomizedSearchCV` |
| Serialization | `pickle` |
| Web App | `streamlit` |

---

## 🔬 ML Pipeline Overview

### 1. Data Integration
Training and test sets are concatenated for unified preprocessing:
```python
df = pd.concat([df_train, df_test])
```

### 2. Missing Value Imputation
Missing values were handled using domain knowledge and statistical reasoning:

| Strategy | Features |
|---|---|
| **Mode** (categorical) | `MSZoning`, `Electrical`, `KitchenQual`, `Functional`, `SaleType` |
| **Median** (skewed numerical) | `LotFrontage` |
| **Constant `"NA"`** (domain knowledge) | `Alley`, `FireplaceQu`, `PoolQC`, `Fence`, `MiscFeature`, all Basement & Garage categoricals |
| **Constant `0`** (absence of feature) | All Basement & Garage numerical features |

### 3. Feature Transformation
Year and month columns were converted from numeric to categorical strings to prevent the model from treating them as continuous ordinal values:
```python
for_num_conv = ["MSSubClass", "YearBuilt", "YearRemodAdd", "GarageYrBlt", "MoSold", "YrSold"]
for feat in for_num_conv:
    df_mvi[feat] = df_mvi[feat].astype(str)

# Month numbers converted to abbreviations (e.g., 6 → "Jun")
df_mvi["MoSold"] = df_mvi["MoSold"].apply(lambda x: calendar.month_abbr[x])
```

### 4. Encoding

**Ordinal Encoding** — applied to 17 quality/condition features where order matters (e.g., `Po < Fa < TA < Gd < Ex`):
```python
df_mvi["KitchenQual"] = df_mvi["KitchenQual"].astype(
    CategoricalDtype(categories=["Po", "Fa", "TA", "Gd", "Ex"], ordered=True)
).cat.codes
```

**One-Hot Encoding** — applied to all remaining nominal categorical columns:
```python
object_features = df_encode.select_dtypes(include="object").columns.tolist()
df_encode = pd.get_dummies(df_encode, columns=object_features, drop_first=True)
```

### 5. Feature Scaling
`StandardScaler` (z-score normalization) was fit **only on training data** and persisted for use during inference:
```python
# Formula: z = (x - μ) / σ
sc = StandardScaler()
sc.fit(X_train)

pickle.dump(sc, open("scaler.pkl", "wb"))
pickle.dump(list(X_train.columns), open("column_names.pkl", "wb"))

X_train = sc.transform(X_train)
X_test = sc.transform(X_test)
```

### 6. Model Selection
Nine regression models were evaluated using **7-Fold Cross-Validation** with R² scoring:

```python
models = {
    "LinearRegression", "SVR", "SGDRegressor",
    "KNeighborsRegressor", "GaussianProcessRegressor",
    "DecisionTreeRegressor", "GradientBoostingRegressor",
    "RandomForestRegressor", "XGBRegressor"
}

def test_model(model):
    cv = KFold(n_splits=7, shuffle=True, random_state=45)
    r2 = make_scorer(r2_score)
    r2_val_score = cross_val_score(model, X_train, y_train, cv=cv, scoring=r2)
    return [r2_val_score.mean()]
```

**GradientBoostingRegressor** achieved the best cross-validated R² and was selected for tuning.

### 7. Hyperparameter Tuning
`RandomizedSearchCV` with 25 iterations was used to efficiently search the hyperparameter space:
```python
param_dist = {
    "n_estimators":      [100, 200, 300, 400, 500],
    "learning_rate":     [0.01, 0.03, 0.05, 0.1],
    "max_depth":         [3, 4, 5],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf":  [1, 2, 4],
    "subsample":         [0.8, 0.9, 1.0]
}

random_search = RandomizedSearchCV(
    gbr, param_distributions=param_dist,
    n_iter=25, cv=cv, scoring="r2",
    n_jobs=-1, random_state=42
)
```

The best estimator was retrained on the full training set and serialized:
```python
pickle.dump(best_model, open("house_model.pkl", "wb"))
```

---

## 🖥️ Running the App

### Prerequisites

```bash
pip install streamlit scikit-learn pandas numpy xgboost
```

### Launch

```bash
streamlit run app.py
```

### How It Works

The app collects user inputs (quality ratings, square footage, bedrooms, etc.) and reconstructs a full-featured row matching the training schema. It then:

1. Applies the same **ordinal encoding** used during training
2. Applies the same **one-hot encoding** via `pd.get_dummies`
3. **Reindexes** the row to match training column order exactly (missing columns filled with `0`)
4. **Scales** the row using the saved `scaler.pkl`
5. Runs inference with `house_model.pkl` and displays the predicted price

```python
# Column alignment (ensures inference matches training schema exactly)
df = df.reindex(columns=column_names, fill_value=0)

# Scale and predict
df_scaled = scaler.transform(df)
predicted_price = model.predict(df_scaled)[0]
```

---

## 📊 User Input Features

| Feature | Type | Description |
|---|---|---|
| Overall Quality | Slider (1–10) | General material and finish quality |
| Overall Condition | Slider (1–10) | General condition of the house |
| Year Built | Number | Original construction year |
| Living Area | Number | Above-grade living area (sq ft) |
| Basement Area | Number | Total basement square footage |
| Garage Size | Number | Garage capacity (no. of cars) |
| Full Bathrooms | Number | Number of full bathrooms |
| Bedrooms | Number | Bedrooms above ground |
| Kitchen Quality | Dropdown | Po / Fa / TA / Gd / Ex |

---

## 📌 Notes

- All non-user-input fields are set to sensible defaults representative of a typical property (e.g., `MSZoning = "RL"`, `Neighborhood = "NAmes"`).
- The model predicts on a **single row**, so inference is near-instant.
- Saved artifacts (`house_model.pkl`, `scaler.pkl`, `column_names.pkl`) must be in the same directory as `app.py`.
