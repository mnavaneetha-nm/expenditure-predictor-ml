from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained ML model
model = joblib.load("best_expenditure_model.pkl")

# Load dataset for dropdown options
df = pd.read_csv("DatasetExp/expense_transactions.csv")

descriptions = sorted(df["Description"].unique())
payment_methods = sorted(df["Payment_Method"].unique())
categories = sorted(df["Category"].unique())


@app.route("/")
def home():
    return render_template(
        "index.html",
        descriptions=descriptions,
        payment_methods=payment_methods,
        categories=categories
    )


@app.route("/predict", methods=["POST"])
def predict():

    date_value = request.form["date"]
    description = request.form["description"]
    payment_method = request.form["payment_method"]
    category = request.form["category"]

    # Convert selected date
    selected_date = pd.to_datetime(date_value)

    # Prepare input exactly like training data
    input_data = pd.DataFrame({
        "Description": [description],
        "Payment_Method": [payment_method],
        "Category": [category],
        "Year": [selected_date.year],
        "Month": [selected_date.month],
        "Day": [selected_date.day],
        "DayOfWeek": [selected_date.dayofweek]
    })

    # Predict expenditure
    prediction = model.predict(input_data)[0]

    prediction = round(prediction, 2)
    return render_template(

    "index.html",
    prediction=f"{prediction:,.2f}",
    descriptions=descriptions,
    payment_methods=payment_methods,
    categories=categories,
    selected_date=date_value,
    selected_description=description,
    selected_payment=payment_method,
    selected_category=category
)
@app.route("/dashboard")
def dashboard():

    # Load dataset
    data = pd.read_csv("DatasetExp/expense_transactions.csv")

    # Total expenditure
    total_expenditure = round(data["Amount"].sum(), 2)

    # Average expenditure
    average_expenditure = round(data["Amount"].mean(), 2)

    # Category-wise expenditure
    category_summary = (
        data.groupby("Category")["Amount"]
        .sum()
        .sort_values(ascending=False)
    )

    category_labels = category_summary.index.tolist()
    category_values = category_summary.values.tolist()

    # Highest spending category
    highest_category = category_summary.index[0]

    # Convert Date column
    data["Date"] = pd.to_datetime(data["Date"])

    # Monthly expenditure
    monthly_summary = (
        data.groupby(data["Date"].dt.to_period("M"))["Amount"]
        .sum()
    )

    monthly_labels = [
        str(month) for month in monthly_summary.index
    ]

    monthly_values = monthly_summary.values.tolist()

    # Send everything to dashboard.html
    return render_template(
        "dashboard.html",
        total_expenditure=total_expenditure,
        average_expenditure=average_expenditure,
        highest_category=highest_category,
        category_labels=category_labels,
        category_values=category_values,
        monthly_labels=monthly_labels,
        monthly_values=monthly_values
    )

if __name__ == "__main__":
    app.run(debug=True)