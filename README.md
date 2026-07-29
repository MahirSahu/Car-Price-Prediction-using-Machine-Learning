# 🚗 Car Price Prediction using Machine Learning

## Overview

This project predicts the selling price of used cars using **Linear Regression**. The dataset is cleaned, preprocessed, and transformed before training the model. Categorical features are encoded using **One-Hot Encoding**, and the best train-test split is automatically selected by evaluating multiple random states based on the **R² Score**.

The trained model is finally saved using **Pickle**, allowing it to be reused without retraining.

---

## Features

- Data Cleaning and Preprocessing
- Removal of invalid and missing values
- Price and distance conversion to numeric format
- Feature Engineering
- One-Hot Encoding of categorical variables
- Linear Regression Model
- Automatic Best Random State Selection
- Model Evaluation using R² Score
- Save Trained Model (.pkl)
- Predict price of new cars

---

## Dataset

The dataset contains information about used cars including:

- Car Name
- Company
- Manufacturing Year
- Kilometers Driven
- Fuel Type
- Price (Target Variable)

---

## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Pickle

---

## Project Workflow

### 1. Load Dataset

The dataset is loaded using Pandas.

```python
pd.read_csv()
```

---

### 2. Data Cleaning

The following preprocessing steps are performed:

- Remove invalid year values
- Convert year into integer
- Remove "Ask For Price" records
- Convert price into integer
- Clean kilometers driven column
- Remove missing fuel type values
- Shorten long car names
- Remove extreme price outliers
- Reset dataframe index

The cleaned dataset is saved as

```
Cleaned Data.csv
```

---

### 3. Feature Selection

Input Features

- Name
- Company
- Year
- Kilometers Driven
- Fuel Type

Target

- Price

---

### 4. Encoding

Categorical variables are encoded using

- OneHotEncoder

The remaining numeric columns are passed without modification.

---

### 5. Model Training

The project uses

```
Linear Regression
```

A pipeline combines preprocessing and model training.

---

### 6. Best Random State Selection

Instead of selecting a random train-test split manually, the program tests **1000 different random states**.

For each random state:

- Train model
- Predict prices
- Calculate R² Score

The random state producing the highest R² Score is selected automatically.

Example:

```
Best Random State : 324
Best R² Score : 0.89
```

*(The actual values depend on the dataset.)*

---

### 7. Final Model

The model is retrained using the best random state and evaluated again.

Example Output

```
Final R² Score : 0.89
```

---

### 8. Save Model

The trained model is saved as

```
LinearRegressionModel.pkl
```

This model can later be loaded directly for prediction.

---

### 9. Prediction

Example prediction:

```python
pipe.predict(
    pd.DataFrame(
        [["Maruti Suzuki Swift",
          "Maruti",
          2019,
          100,
          "Petrol"]],
        columns=[
            "name",
            "company",
            "year",
            "kms_driven",
            "fuel_type"
        ]
    )
)
```

Output:

```
Predicted Price : ₹ XXXXXXX
```

(The predicted value depends on the trained model.)

---

# Output of the Program

After execution, the program produces the following outputs:

### Console Output

```
Best Random State : <best_random_state>

Best R² Score : <highest_r2_score>

Final R² Score : <final_r2_score>

Model saved successfully!

Predicted Price : <predicted_price>
```

Example:

```
Best Random State : 324

Best R² Score : 0.8923

Final R² Score : 0.8923

Model saved successfully!

Predicted Price : 546782.41
```

*(Numbers vary depending on the dataset and train-test split.)*

---

# Files Generated

After running the script, the following files are created:

```
Cleaned Data.csv
```

Contains the cleaned dataset.

```
LinearRegressionModel.pkl
```

Serialized trained machine learning model.

---

# Project Structure

```
Car-Price-Prediction/
│
├── quikr_car.csv
├── Cleaned Data.csv
├── LinearRegressionModel.pkl
├── car_price_prediction.py
├── README.md
└── requirements.txt
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Car-Price-Prediction.git
```

Move into the project directory

```bash
cd Car-Price-Prediction
```

Install dependencies

```bash
pip install numpy pandas scikit-learn
```

Run the project

```bash
python car_price_prediction.py
```

---

# Future Improvements

- Random Forest Regressor
- Decision Tree Regressor
- XGBoost Regressor
- Feature Scaling
- Hyperparameter Tuning
- Streamlit Web Application
- Flask/Django API Deployment

---

# Learning Outcomes

This project demonstrates:

- Data Cleaning
- Feature Engineering
- Pipeline Construction
- One-Hot Encoding
- Linear Regression
- Model Evaluation
- Machine Learning Workflow
- Model Serialization using Pickle

---

# Author

**Mahir Sahu**
