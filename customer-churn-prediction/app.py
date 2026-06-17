import joblib
import pandas as pd
import streamlit as st


model = joblib.load("models/final_model.pkl")
X_train_columns = pd.read_csv("data/X_train.csv").columns


def prepare_input(user_data):
    input_df = pd.DataFrame([user_data])

    input_encoded = pd.get_dummies(input_df, drop_first=True)

    final_input = pd.DataFrame(
        0,
        index=[0],
        columns=X_train_columns
    )

    for column in input_encoded.columns:
        if column in final_input.columns:
            final_input[column] = input_encoded[column].values

    return final_input


st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("Customer Churn Prediction")
st.write("Predict whether a telecom customer is likely to leave the company.")

st.sidebar.header("Customer Information")

gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
senior_citizen = st.sidebar.selectbox("Senior Citizen", [0, 1])
partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])

tenure = st.sidebar.slider("Tenure in months", 0, 72, 12)

phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.sidebar.selectbox(
    "Multiple Lines",
    ["No", "Yes", "No phone service"]
)

internet_service = st.sidebar.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

online_security = st.sidebar.selectbox(
    "Online Security",
    ["No", "Yes", "No internet service"]
)

online_backup = st.sidebar.selectbox(
    "Online Backup",
    ["No", "Yes", "No internet service"]
)

device_protection = st.sidebar.selectbox(
    "Device Protection",
    ["No", "Yes", "No internet service"]
)

tech_support = st.sidebar.selectbox(
    "Tech Support",
    ["No", "Yes", "No internet service"]
)

streaming_tv = st.sidebar.selectbox(
    "Streaming TV",
    ["No", "Yes", "No internet service"]
)

streaming_movies = st.sidebar.selectbox(
    "Streaming Movies",
    ["No", "Yes", "No internet service"]
)

contract = st.sidebar.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])

payment_method = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

monthly_charges = st.sidebar.number_input(
    "Monthly Charges",
    min_value=0.0,
    max_value=200.0,
    value=70.0
)

total_charges = st.sidebar.number_input(
    "Total Charges",
    min_value=0.0,
    max_value=10000.0,
    value=1000.0
)

user_data = {
    "gender": gender,
    "SeniorCitizen": senior_citizen,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone_service,
    "MultipleLines": multiple_lines,
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contract,
    "PaperlessBilling": paperless_billing,
    "PaymentMethod": payment_method,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges
}

st.subheader("Customer Profile")

profile_df = pd.DataFrame([user_data])
st.dataframe(profile_df, use_container_width=True)

if st.button("Predict Churn Risk", type="primary"):
    final_input = prepare_input(user_data)

    prediction = model.predict(final_input)[0]
    probability = model.predict_proba(final_input)[0][1]

    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Churn Probability", f"{probability * 100:.2f}%")

    with col2:
        if prediction == 1:
            st.error("High Risk: This customer is likely to churn.")
        else:
            st.success("Low Risk: This customer is likely to stay.")

    st.subheader("Business Interpretation")

    if probability >= 0.7:
        st.write(
            "This customer has a high probability of leaving. "
            "The company should consider retention actions such as a personalized offer, discount, or customer support follow-up."
        )
    elif probability >= 0.4:
        st.write(
            "This customer has a medium churn risk. "
            "The company should monitor this customer and improve engagement."
        )
    else:
        st.write(
            "This customer has a low churn risk. "
            "No immediate retention action is required."
        )
