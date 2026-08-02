import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# -------------------------------
# 1. Load Dataset
# -------------------------------

df = pd.read_csv("DatasetExp/expense_transactions.csv")

print("Dataset loaded successfully!")


# -------------------------------
# 2. Convert Date
# -------------------------------

df["Date"] = pd.to_datetime(df["Date"])

df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["DayOfWeek"] = df["Date"].dt.dayofweek


# -------------------------------
# 3. Features and Target
# -------------------------------

X = df[
    [
        "Description",
        "Payment_Method",
        "Category",
        "Year",
        "Month",
        "Day",
        "DayOfWeek"
    ]
]

y = df["Amount"]


# -------------------------------
# 4. Train-Test Split
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# -------------------------------
# 5. One-Hot Encoding
# -------------------------------

categorical_columns = [
    "Description",
    "Payment_Method",
    "Category"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        )
    ],
    remainder="passthrough"
)


# -------------------------------
# 6. Models
# -------------------------------

models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        random_state=42
    )
}


# -------------------------------
# 7. Train and Evaluate
# -------------------------------

results = []

for name, regressor in models.items():

    print("\nTraining:", name)

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor)
        ]
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    mse = mean_squared_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_test,
        predictions
    )

    results.append(
        {
            "Model": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2 Score": r2
        }
    )


# -------------------------------
# 8. Display Results
# -------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="R2 Score",
    ascending=False
)

print("\n")
print("=" * 60)

print("MODEL COMPARISON RESULTS")

print("=" * 60)

print(
    results_df.to_string(
        index=False
    )
)

print("=" * 60)

best_model = results_df.iloc[0]

print("\nBEST MODEL:")

print(best_model["Model"])

print(
    "R2 Score:",
    round(
        best_model["R2 Score"],
        4
    )
)
import joblib

# Train the best model again
best_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "regressor",
            GradientBoostingRegressor(
                random_state=42
            )
        )
    ]
)

best_pipeline.fit(X_train, y_train)

# Save best model
joblib.dump(
    best_pipeline,
    "best_expenditure_model.pkl"
)

print("\nBest Gradient Boosting model saved successfully!")