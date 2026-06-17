# Customer Churn Prediction

Customer Churn Prediction is a machine learning project that predicts whether a telecom customer is likely to leave the company.

The project covers the full data science workflow: business understanding, data exploration, data cleaning, visualization, preprocessing, model training, model evaluation, model comparison, and deployment with Streamlit.

## Business Problem

Customer churn is a major challenge for subscription-based companies. When customers leave, the company loses revenue and must spend more money to acquire new customers.

The goal of this project is to identify customers who are likely to churn so the company can take retention actions such as personalized offers, discounts, or customer support follow-ups.

## Dataset

The project uses the Telco Customer Churn dataset.

The dataset contains customer information such as:

- Demographic information
- Services subscribed by the customer
- Contract type
- Payment method
- Monthly charges
- Total charges
- Churn status

The target variable is: Churn where : 
0 = customer stayed
1 = customer churned

## Project Workflow

Business problem understanding
Data loading and exploration
Data cleaning
Exploratory data analysis and visualization
Feature preprocessing
Model training
Model evaluation
Model comparison
Streamlit application deployment

## Technologies Used

Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
Streamlit
Joblib

## Exploratory Data Analysis

During the analysis, several churn patterns were identified:
Customers with month-to-month contracts are more likely to  churn.
Customers with low tenure are more likely to churn.
Customers using electronic check as payment method show higher churn.
Monthly charges appear to be related to churn behavior.
The dataset is imbalanced, with more customers who stayed than customers who churned.

## Data Preprocessing

The preprocessing steps included:
Converting TotalCharges from text to numeric format
Handling missing values
Removing the customerID column
Encoding the target variable Churn into 0 and 1
Applying one-hot encoding to categorical variables
Splitting the data into training and testing sets

## Models Used

Two classification models were trained and compared:
Logistic Regression
Random Forest Classifier
Model Performance
Model	Accuracy	Precision	Recall	F1-score	ROC AUC
Logistic Regression	0.801	0.642	0.567	0.602	0.836
Random Forest	0.768	0.553	0.668	0.605	0.820

## Final Model Selection

Random Forest was selected as the final model because it achieved a higher recall than Logistic Regression.

In customer churn prediction, recall is especially important because the business objective is to identify as many at-risk customers as possible before they leave.

## Streamlit Application

The project includes a Streamlit web application where the user can enter customer information and get:
Churn prediction
Churn probability
Risk level
Business interpretation

## How To Run The Project

Install the required dependencies:  pip install -r requirements.txt
Run the Streamlit application:  streamlit run app.py

## Example Prediction

A customer with:
Month-to-month contract
Fiber optic internet
Electronic check payment
Low tenure
High monthly charges
is more likely to have a high churn probability.

## Future Improvements

Tune model hyperparameters
Test additional models such as XGBoost
Add feature scaling for Logistic Regression
Add SHAP explanations for model interpretability
Deploy the Streamlit app online
Add batch prediction from CSV files

## Author
AYA CHKHIT
Data & AI Engineering Student
Seeking a Summer Internship in Data & AI
