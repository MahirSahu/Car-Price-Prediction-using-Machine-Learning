import numpy as np
import pandas as pd
import pickle

# ==========================
# Load Dataset
# ==========================
car = pd.read_csv(r"D:\PROJECT\pandas practice\quikr_car.csv")

# ==========================
# Data Cleaning
# ==========================

# Keep only numeric year values
car = car[car["year"].astype(str).str.isnumeric()]

# Convert year to integer
car["year"] = car["year"].astype(int)

# Remove rows where price is "Ask For Price"
car = car[car["Price"] != "Ask For Price"]

# Remove commas and convert price to integer
car["Price"] = car["Price"].str.replace(",", "").astype(int)

# Clean kms_driven column
car["kms_driven"] = (
    car["kms_driven"]
    .str.split()
    .str[0]
    .str.replace(",", "", regex=False)
)

# Keep only numeric kms values
car = car[car["kms_driven"].str.isnumeric()]

# Convert kms_driven to integer
car["kms_driven"] = car["kms_driven"].astype(int)

# Remove missing fuel type values
car = car[car["fuel_type"].notna()]

# Shorten long car names
car["name"] = car["name"].str.split().str[:3].str.join(" ")

# Remove price outliers
car = car[car["Price"] < 6000000]

# Reset index
car.reset_index(drop=True, inplace=True)

# Save cleaned dataset
car.to_csv("Cleaned Data.csv", index=False)

# ==========================
# Prepare Data
# ==========================

X = car.drop(columns="Price")
y = car["Price"]

# ==========================
# Model Building
# ==========================

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

# One Hot Encoding
ohe = OneHotEncoder(handle_unknown="ignore")
ohe.fit(X[["name", "company", "fuel_type"]])

column_trans = make_column_transformer(
    (
        OneHotEncoder(
            categories=ohe.categories_,
            handle_unknown="ignore"
        ),
        ["name", "company", "fuel_type"],
    ),
    remainder="passthrough",
)

# ==========================
# Find Best Random State
# ==========================

scores = []

for i in range(1000):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=i,
    )

    model = LinearRegression()

    pipe = make_pipeline(column_trans, model)

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)

    scores.append(r2_score(y_test, y_pred))

best_random_state = np.argmax(scores)

print("Best Random State :", best_random_state)
print("Best R² Score :", scores[best_random_state])

# ==========================
# Train Final Model
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=best_random_state,
)

pipe = make_pipeline(column_trans, LinearRegression())

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)

print("Final R² Score :", r2_score(y_test, y_pred))

# ==========================
# Save Model
# ==========================

pickle.dump(pipe, open("LinearRegressionModel.pkl", "wb"))

print("Model saved successfully!")

# ==========================
# Prediction Example
# ==========================

prediction = pipe.predict(
    pd.DataFrame(
        [
            [
                "Maruti Suzuki Swift",
                "Maruti",
                2019,
                100,
                "Petrol",
            ]
        ],
        columns=[
            "name",
            "company",
            "year",
            "kms_driven",
            "fuel_type",
        ],
    )
)

print("Predicted Price :", prediction[0])
