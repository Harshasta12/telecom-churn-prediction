import streamlit as st
import pandas as pd
import joblib
import json

# ---- Load saved artifacts (same pattern as your deploy.py) ----
@st.cache_resource
def load_artifacts():
    model = joblib.load('churn_model_xgb.pkl')
    scaler = joblib.load('churn_scaler.pkl')
    with open('model_config.json', 'r') as f:
        config = json.load(f)
    return model, scaler, config

model, scaler, config = load_artifacts()
threshold = config['threshold']

# ---- Page setup ----
st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")
st.title("📉 Customer Churn Predictor")
st.write(
    "Predicts the likelihood a telecom customer will churn, using an XGBoost model "
    "tuned to prioritize catching at-risk customers (recall) over minimizing false alarms."
)

st.divider()

# ---- Input form ----
st.subheader("Customer Details")

col1, col2 = st.columns(2)

with col1:
    account_length = st.number_input("Account length (days)", min_value=0, max_value=500, value=100)
    area_code = st.selectbox("Area code", [408, 415, 510])
    international_plan = st.radio("International plan", ["No", "Yes"])
    voice_mail_plan = st.radio("Voice mail plan", ["No", "Yes"])
    customer_service_calls = st.number_input("Customer service calls", min_value=0, max_value=15, value=1)

with col2:
    total_day_calls = st.number_input("Total day calls", min_value=0, max_value=200, value=100)
    total_day_charge = st.number_input("Total day charge ($)", min_value=0.0, max_value=100.0, value=30.0)
    total_eve_calls = st.number_input("Total evening calls", min_value=0, max_value=200, value=100)
    total_eve_charge = st.number_input("Total evening charge ($)", min_value=0.0, max_value=50.0, value=17.0)
    total_night_calls = st.number_input("Total night calls", min_value=0, max_value=200, value=100)

col3, col4 = st.columns(2)
with col3:
    total_night_charge = st.number_input("Total night charge ($)", min_value=0.0, max_value=30.0, value=9.0)
    total_intl_calls = st.number_input("Total international calls", min_value=0, max_value=20, value=4)
with col4:
    total_intl_charge = st.number_input("Total international charge ($)", min_value=0.0, max_value=10.0, value=2.7)

st.divider()

# ---- Predict button ----
if st.button("Predict Churn Risk", type="primary"):

    customer = {
        'account length': account_length,
        'area code': area_code,
        'international plan': 1 if international_plan == "Yes" else 0,
        'voice mail plan': 1 if voice_mail_plan == "Yes" else 0,
        'total day calls': total_day_calls,
        'total day charge': total_day_charge,
        'total eve calls': total_eve_calls,
        'total eve charge': total_eve_charge,
        'total night calls': total_night_calls,
        'total night charge': total_night_charge,
        'total intl calls': total_intl_calls,
        'total intl charge': total_intl_charge,
        'customer service calls': customer_service_calls,
    }

    # Ensure column order matches training exactly
    customer_df = pd.DataFrame([customer])[config['features']]
    scaled = scaler.transform(customer_df)
    probability = model.predict_proba(scaled)[:, 1][0]
    prediction = int(probability >= threshold)

    st.subheader("Result")

    if prediction == 1:
        st.error(f"⚠️ High churn risk — predicted probability: **{probability:.1%}**")
        st.write("This customer is flagged for retention outreach.")
    else:
        st.success(f"✅ Low churn risk — predicted probability: **{probability:.1%}**")
        st.write("This customer is not currently flagged as at-risk.")

    st.caption(f"Decision threshold: {threshold} (tuned for higher recall — catching more true churners at the cost of some false alarms)")

    with st.expander("See raw model output"):
        st.json({"churn_probability": float(probability), "threshold_used": threshold, "prediction": prediction})

st.divider()
st.caption("Model: XGBoost | Trained on the 'Churn in Telecoms' dataset | Threshold tuned via precision-recall analysis to prioritize recall")