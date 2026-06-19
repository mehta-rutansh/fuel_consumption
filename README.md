# ⛽ Fuel Consumption Analysis for Automobiles

A data science project that analyzes vehicle fuel consumption and CO₂ emissions, trains a machine learning model to predict emissions, and presents everything through an interactive Streamlit dashboard — including a bulk CSV scanner that can predict emissions for any uploaded vehicle file.

---

## 📌 Abstract

This project studies how vehicle attributes — engine size, cylinders, transmission, fuel type, and vehicle class — affect fuel consumption and CO₂ emissions. Using a dataset of 22,556 vehicles spanning 2000–2022, it covers data cleaning, exploratory data analysis (EDA), feature engineering, machine learning model training, and an interactive web app for predictions, including bulk predictions on user-uploaded files.

---

## 📊 Dataset

| Property | Value |
|---|---|
| Rows | 22,556 |
| Columns | 13 |
| Years covered | 2000 – 2022 |
| Car makes | 87 |
| Missing values | None |
| Source file | `Fuel_Consumption.csv` |

**Columns:** `YEAR, MAKE, MODEL, VEHICLE CLASS, ENGINE SIZE, CYLINDERS, TRANSMISSION, FUEL, FUEL CONSUMPTION (city), HWY (L/100 km), COMB (L/100 km), COMB (mpg), EMISSIONS`

**Target variable:** `EMISSIONS` (CO₂ in g/km)

---

## 🗂️ Project Structure

```
fuel_consumption/
├── Fuel_Consumption.csv          ← original dataset
├── fuel_clean.csv                ← after cleaning (step 1)
├── fuel_engineered.csv           ← after feature engineering (step 3)
├── fuel_model.pkl                ← trained Random Forest model
├── step1_data_loading.py         ← load + clean data
├── step2_eda.py                  ← exploratory analysis + plots
├── step3_feature_engineering.py  ← new features + encoding
├── step4_model_training.py       ← train & compare models
├── app.py                        ← Streamlit dashboard (Home, EDA, Predict, Bulk Scanner)
├── bulk_scanner_page.py          ← bulk CSV/Excel scanner page (pasted into app.py)
├── plot1_emissions_dist.png      ← saved EDA charts
├── plot2_engine_vs_co2.png
├── ... (plot3–plot8)
└── README.md
```

---

## ⚙️ Installation

```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit openpyxl
```

`openpyxl` is required for the Bulk Scanner to read `.xlsx` files.

---

## 🚀 How to Run (in order)

```bash
python step1_data_loading.py          # clean data → fuel_clean.csv
python step2_eda.py                   # generate EDA plots
python step3_feature_engineering.py   # add features → fuel_engineered.csv
python step4_model_training.py        # train model → fuel_model.pkl
streamlit run app.py                  # launch the dashboard
```

The app opens at **http://localhost:8501**

---

## 🔬 Project Stages

### 1. Data Cleaning (`step1_data_loading.py`)
- Loads the raw CSV
- Checks shape, data types, missing values
- Removes duplicate records
- Saves cleaned data to `fuel_clean.csv`

### 2. Exploratory Data Analysis (`step2_eda.py`)
Generates 8 visualizations:
- CO₂ emissions distribution (histogram)
- Engine size vs CO₂ (scatter plot)
- City vs highway fuel consumption (scatter plot)
- Average CO₂ by fuel type (bar chart)
- Fuel type share (pie chart)
- CO₂ by number of cylinders (box plot)
- CO₂ trend by year (line chart)
- Feature correlation (heatmap)

### 3. Feature Engineering (`step3_feature_engineering.py`)
New features created:
| Feature | Meaning |
|---|---|
| `CITY_HWY_AVG` | Average of city and highway consumption |
| `ENGINE_PER_CYL` | Engine size divided by cylinder count |
| `EFF_SCORE` | Fuel efficiency score (100 / combined L/100km) |
| `VEHICLE_AGE` | 2024 − model year |

Categorical columns (`FUEL`, `VEHICLE CLASS`, `TRANSMISSION`) are label-encoded into `FUEL_ENC`, `CLASS_ENC`, `TRANS_ENC`.

### 4. Model Training (`step4_model_training.py`)
Two models are trained and compared on a 80/20 train-test split:

| Model | MAE | R² |
|---|---|---|
| Linear Regression | 11.35 | 0.9163 |
| **Random Forest** | **0.31** | **0.9997** |

**Random Forest is selected and saved** as `fuel_model.pkl` — it predicts CO₂ emissions almost perfectly (off by less than 1 g/km on average).

### 5. Streamlit App (`app.py`)
Four pages:
- **🏠 Home** — KPI cards, dataset preview, key statistics
- **📊 EDA Charts** — interactive tabbed charts (distributions, relationships, categories, heatmap), all redesigned with a dark modern theme and Indian-context explanations
- **🤖 Predict CO₂** — single-vehicle prediction form with km/L, L/100km, and mpg all shown together, Indian vehicle examples (Swift, Creta, Fortuner, etc.) as reference points
- **📁 Bulk Scanner** — upload any CSV/Excel file with car data (any column names), auto-map columns to model features, predict CO₂ for every row, and download results

---

## 📁 Bulk Scanner — How It Works

The Bulk Scanner lets you upload **any** vehicle CSV/Excel file, even if its column names don't match this project's dataset:

1. Upload file → preview is shown
2. App auto-guesses column mapping (e.g. "Car Engine (L)" → Engine Size) using keyword matching
3. You review/correct mappings via dropdowns
4. Missing fields (e.g. no highway consumption) are estimated from what's available
5. Unknown fuel type / vehicle class / transmission values default to safe averages (Regular Petrol, Mid-size, Automatic)
6. Predictions run for every valid row
7. Results table + distribution chart + CSV download

---

## 🧠 Key Insights

- **Engine size and cylinder count are the strongest predictors of CO₂ emissions** (confirmed by the correlation heatmap and feature importance plot)
- CO₂ emissions have **declined over the years** (2000 → 2022) due to better engine technology
- **CNG/Natural Gas** vehicles have the lowest average CO₂; **Diesel/Premium Petrol** vehicles tend to emit more
- Highway driving consistently uses **less fuel** than city driving across all vehicle classes

---

## 🛠️ Tech Stack

- **Python**, **Pandas**, **NumPy** — data handling
- **Matplotlib**, **Seaborn** — visualization
- **Scikit-learn** — Random Forest Regressor, Label Encoding, train/test split
- **Streamlit** — web dashboard
- **Pickle** — model persistence

---

## 👤 Author

Rutansh Mehta — M.Sc. IT, GLS University, Ahmedabad
