# 🏠 House Price Predictor

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)

**An interactive machine learning web app that predicts house sale prices based on key property features.**

[Demo](#demo) · [Features](#features) · [Installation](#installation) · [Usage](#usage) · [Model Details](#model-details) · [Project Structure](#project-structure)

---

</div>

## 📸 Demo

> Enter property details using the intuitive sidebar controls and instantly receive a predicted sale price powered by a trained regression model.

```
🏡  Overall Quality:   ████████░░  8/10
📐  Living Area:       1,850 sq ft
🛏️  Bedrooms:          3
🚿  Full Bathrooms:    2
🚗  Garage Size:       2 cars

  → Estimated Sale Price:  $214,500
```

---

## ✨ Features

- **Interactive UI** — Sliders, dropdowns, and number inputs for a seamless experience
- **Real-time prediction** — Instantly estimates sale price with a single button click
- **Robust preprocessing** — Ordinal encoding + one-hot encoding + feature scaling, matching the training pipeline exactly
- **Training-aligned inference** — Uses saved `scaler.pkl` and `column_names.pkl` to guarantee consistent feature alignment
- **Handles edge cases** — Missing dummy columns are filled with `0`; extra columns are dropped automatically

---

## 🧩 Input Features

| Feature | Type | Description |
|---|---|---|
| `OverallQual` | Slider (1–10) | Overall material and finish quality |
| `OverallCond` | Slider (1–10) | Overall condition of the house |
| `YearBuilt` | Number | Original construction year |
| `GrLivArea` | Number | Above-grade living area (sq ft) |
| `TotalBsmtSF` | Number | Total basement area (sq ft) |
| `GarageCars` | Number | Garage capacity (number of cars) |
| `FullBath` | Number | Full bathrooms above grade |
| `BedroomAbvGr` | Number | Bedrooms above grade |
| `KitchenQual` | Dropdown | Kitchen quality (Po / Fa / TA / Gd / Ex) |

---

## 🗂️ Project Structure

```
house-price-predictor/
│
├── app.py                  # Streamlit app — UI + inference pipeline
├── house_model.pkl         # Trained regression model
├── scaler.pkl              # Fitted StandardScaler (from training)
├── column_names.pkl        # Ordered feature column list (from training)
│
├── notebook/
│   └── training.ipynb      # Model training, EDA, and preprocessing notebook
│
└── README.md
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.8 or higher
- pip

### 1. Clone the repository

```bash
git clone https://github.com/your-username/house-price-predictor.git
cd house-price-predictor
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt`**
```
streamlit
numpy
pandas
scikit-learn
```

### 3. Ensure model files are present

Make sure the following files are in the root directory:

```
house_model.pkl
scaler.pkl
column_names.pkl
```

> ⚠️ These files are generated during model training. If missing, run the training notebook first.

---

## 🚀 Usage

```bash
streamlit run app.py
```

Then open your browser at **`http://localhost:8501`**.

### Steps

1. Adjust the sliders and inputs to describe your property.
2. Click **"Predict Price"**.
3. The estimated sale price will appear as a green success banner.

---

## 🤖 Model Details

### Training Data
The model was trained on the [Ames Housing Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques), which contains 79 explanatory variables describing residential homes in Ames, Iowa.

### Preprocessing Pipeline

```
Raw Input
    │
    ▼
Ordinal Encoding       ← ExterQual, KitchenQual, BsmtQual, GarageFinish, etc.
    │
    ▼
One-Hot Encoding       ← All remaining categorical/object columns
    │
    ▼
Column Alignment       ← Reindex to training column order (fill missing = 0)
    │
    ▼
Standard Scaling       ← StandardScaler fitted on training data
    │
    ▼
Model Prediction       ← Trained regression model (house_model.pkl)
    │
    ▼
Predicted Sale Price   💰
```

### Ordinal Features Encoded

| Column | Order (low → high) |
|---|---|
| `ExterQual`, `ExterCond`, `HeatingQC`, `KitchenQual` | Po → Fa → TA → Gd → Ex |
| `BsmtQual`, `BsmtCond` | NA → Po → Fa → TA → Gd → Ex |
| `GarageFinish` | NA → Unf → RFn → Fin |
| `BsmtExposure` | NA → No → Mn → Av → Gd |
| `PavedDrive` | N → P → Y |
| `Functional` | Sal → Sev → Maj2 → … → Typ |

---

## 🔧 How the Inference Pipeline Stays Aligned with Training

A common pitfall in ML deployment is **feature mismatch** between training and inference. This app avoids it with two key fixes:

```python
# 1. Reindex columns to exactly match training order
df = df.reindex(columns=column_names, fill_value=0)

# 2. Scale with the same fitted scaler from training
df_scaled = scaler.transform(df)
```

This ensures the model always receives features in the **exact same shape and scale** it was trained on.

---

## 📊 Example Predictions

| Quality | Area (sq ft) | Year Built | Bedrooms | Predicted Price |
|---|---|---|---|---|
| 5 | 1,200 | 1985 | 3 | ~$120,000 |
| 7 | 1,800 | 2000 | 3 | ~$185,000 |
| 9 | 2,500 | 2005 | 4 | ~$295,000 |

> *These are illustrative estimates. Actual predictions depend on the trained model.*

---

## 🙌 Acknowledgements

- [Ames Housing Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) by Dean De Cock
- [Streamlit](https://streamlit.io/) for the rapid web app framework
- [scikit-learn](https://scikit-learn.org/) for model training and scaling utilities

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
