# 💰 Expenditure Predictor Using Machine Learning

## 📌 Project Overview
The Expenditure Predictor is a Machine Learning based web application that predicts transaction expenditure based on user-provided details.

The application also provides an Expenditure Dashboard to display overall spending information.

## 🎯 Features
- Predicts expenditure for a transaction
- User-friendly web interface
- Expenditure dashboard
- Displays total expenditure
- Displays average transaction amount
- Identifies highest spending category
- Compares multiple Machine Learning regression algorithms

## 🤖 Machine Learning Models
The following regression models were compared:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor

Gradient Boosting achieved the best performance among the tested models and was selected for expenditure prediction.

## 📊 Model Evaluation
Models were evaluated using:

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score

Best Model: **Gradient Boosting Regressor**

R² Score: **0.5731**

## 📂 Dataset Features
The dataset contains:

- Date
- Description
- Amount
- Payment Method
- Category

`Amount` is used as the target variable for prediction.

## 🛠️ Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Flask
- HTML
- CSS
- Joblib

## 🌐 Web Application
The Flask backend connects the trained Machine Learning model with an HTML/CSS frontend.

Users provide transaction information and the trained model predicts the expected expenditure.

## 📈 Dashboard
The dashboard provides:

- Total Expenditure
- Average Transaction
- Highest Spending Category

## ▶️ How to Run

Install the required packages:

pip install -r requirements.txt

Run the application:

python app.py

Then open the local Flask address shown in the terminal.

## 👩‍💻 Project Type
Machine Learning Regression + Flask Web Application
## 🌐 Live Demo

🚀 [Click here to open Expenditure Predictor](https://expenditure-predictor-ml.onrender.com)
