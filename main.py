import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import numpy as np

# Load dataset
df = pd.read_csv("DatasetExp/expense_transactions.csv")

print("Dataset loaded successfully!")

# Convert Date
df["Date"] = pd.to_datetime(df["Date"])

# Extract useful date features
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["DayOfWeek"] = df["Date"].dt.dayofweek

# Features used to predict expenditure
X = df[
    [
        "Description",
        "Payment_Method",
        "Category",
        "Year",
        "Month",
        "Day",
        "DayOfWeek",
    ]
]

# Target
y = df["Amount"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Categorical columns
categorical_features = [
    "Description",
    "Payment_Method",
    "Category"
]

# Encoder
preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)

# Random Forest model
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

print("Training model...")

# Train
model.fit(X_train, y_train)

print("Model training completed!")

# Prediction
predictions = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print("\nMODEL RESULTS")
print("-------------------------")
print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

# Save model
joblib.dump(model, "expenditure_model.pkl")

print("\nModel saved successfully as expenditure_model.pkl")