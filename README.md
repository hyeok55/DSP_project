Here is your full project description rewritten clearly and professionally in **English**:

---

# **Heat Treatment Process Quality Prediction System**

## **Dataset**

[https: *[Google Drive link]*](https://drive.google.com/drive/folders/1vA4kwzpzHsanL1XRTmv8xGbY2IizjkHz?usp=drive_link)

A system for predicting defect rates using manufacturing process data analysis and time-series forecasting models.

---

# **Project Structure**

```
src/
├── data/                    # Raw data files
├── models/                  # Model definitions (DLinear, NLinear, LSTM)
├── onnx/                    # Trained ONNX models
├── analyze.ipynb            # Step 1: Statistical analysis
├── predict.ipynb            # Step 2: Time-series model training
├── dashboard.ipynb          # Step 3: Real-time monitoring dashboard
├── requirements.txt         # Required packages
└── result.csv               # Final model performance results
```

---

# **Environment Setup**

```bash
pip install -r src/requirements.txt
```

**Key packages**

* Python 3.8+
* pandas, numpy, scipy
* scikit-learn, torch
* onnxruntime
* matplotlib, flask

---

# **Execution Workflow**

## **1. Statistical Analysis (`analyze.ipynb`)**

```bash
jupyter notebook src/analyze.ipynb
```

Tasks:

* Analyze 37 process variables
* Select top 3 key variables
  (using Cohen’s d, t-test, Mann–Whitney U test)
* Output: `src/data/analyze.csv`

---

## **2. Model Training (`predict.ipynb`)**

```bash
jupyter notebook src/predict.ipynb
```

Tasks:

* Train time-series forecasting models (DLinear, NLinear, LSTM)
* Perform 36 total experiments
  (3 models × 3 targets × 2 learning rates × 2 loss functions)
* Output:

  * `src/onnx/*.onnx`
  * `src/result.csv`

---

# **Key Results**

## **Statistical Analysis**

* Selected key variables:

  * Quenching Furnace Temperature – Zone 1 (Min)
  * Drying Furnace Temperature – Zone 2 (Min)
  * Quenching Furnace Temperature – Zone 4 (Min)
* Cohen’s d: **1.68–2.31** (large effect size)

---

## **Best Model Performance**

| Target Variable                | Model | MAE   | RMSE  | SMAPE(%) |
| ------------------------------ | ----- | ----- | ----- | -------- |
| Drying Furnace Temp. Zone 2    | LSTM  | 0.334 | 0.563 | 98.15    |
| Quenching Furnace Temp. Zone 1 | LSTM  | 0.325 | 0.669 | 74.93    |
| Quenching Furnace Temp. Zone 4 | LSTM  | 0.312 | 0.456 | 115.97   |

**Optimal Hyperparameters:**
Learning Rate **0.01** + **L1 Loss**

---

If you want, I can also convert this into:
✅ A README.md
✅ A technical report format
✅ A PPT summary
Just tell me what you need!
