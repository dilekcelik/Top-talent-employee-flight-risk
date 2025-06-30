import streamlit as st
import pickle
import pandas as pd
from PIL import Image
import shap
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

# --- Page Configuration ---
st.set_page_config(page_title="Flight Risk Employee Prediction", layout="wide")

# --- Custom CSS ---
st.markdown("""
    <style>
    .title {
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        color: #2C3E50;
    }
    .info-box {
        background-color: #ECF0F1;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .prediction-box {
        background-color: #D6EAF8;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #2980B9;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .prediction-title {
        font-size: 24px;
        font-weight: bold;
        color: #65d2f0;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🧠 Flight Risk Employee Prediction</div>', unsafe_allow_html=True)

# --- Model Info ---
with st.expander("ℹ️ Model Info", expanded=True):
    st.markdown("""
        <div class="info-box">
        <b>Recall and Accuracy Scores:</b><br>
        • <b>XGBoost:</b> Recall (1) = 0.91, Accuracy = 0.98
        </div>
    """, unsafe_allow_html=True)

# --- Sidebar Input ---
st.sidebar.subheader("🎛️ Intelligent Employee Retention System for Churn Prediction")
satisfaction_level = st.sidebar.slider("Satisfaction Level", 0, 10)
last_evaluation = st.sidebar.slider("Last Evaluation", 0, 10)
number_project = st.sidebar.slider("Number Project", 2, 7)
average_montly_hours = st.sidebar.slider("Average Montly Hours", 0, 500, step=8)
time_spend_company = st.sidebar.slider("Time Spend Company", 1, 10)
Work_accident = st.sidebar.slider("Work Accident", 0, 1)
promotion_last_5years = st.sidebar.slider("Promotion in Last 5 Years", 0, 1)
Departments = st.sidebar.radio("Departments", ("sales", "IT", "RandD", "Departments_hr", "mng", "support", "technical"))
salary = st.sidebar.radio("Salary", ("low", "medium", "high"))
model_name = st.selectbox("Select your model:", ("XGB Model",))

# --- User Input to DataFrame ---
input_data = pd.DataFrame([{
    'satisfaction_level': satisfaction_level / 10,
    'last_evaluation': last_evaluation / 10,
    'number_project': number_project,
    'average_montly_hours': average_montly_hours,
    'time_spend_company': time_spend_company,
    'Work_accident': Work_accident,
    'promotion_last_5years': promotion_last_5years,
    'salary': salary,
    'Departments': Departments
}])

st.subheader("📋 Selected Configuration")
st.dataframe(input_data)

# --- Preprocessing ---
input_data['salary'] = input_data['salary'].map({'low': 0, 'medium': 1, 'high': 2})
columns = ['satisfaction_level', 'last_evaluation', 'number_project',
           'average_montly_hours', 'time_spend_company', 'Work_accident',
           'promotion_last_5years', 'salary', 'Departments_IT',
           'Departments_RandD', 'Departments_accounting', 'Departments_hr',
           'Departments_management', 'Departments_marketing',
           'Departments_product_mng', 'Departments_sales', 'Departments_support',
           'Departments_technical']
input_data = pd.get_dummies(input_data).reindex(columns=columns, fill_value=0)

scaler_churn = pickle.load(open("scaler_churn", "rb"))
scaled_input = scaler_churn.transform(input_data)
st.subheader("⚙️ Scaled Features")
st.dataframe(pd.DataFrame(scaled_input, columns=columns))

# --- Load Model ---
model = pickle.load(open("XGB_model", "rb"))

# --- Prediction Section ---
st.markdown('<div class="prediction-box"><div class="prediction-title">🎯 Prediction Result</div>', unsafe_allow_html=True)
if st.button("Predict Churn"):
    prediction = model.predict(scaled_input)
    if prediction[0] == 1:
        st.markdown('<p style="color: red; font-weight: bold;">🚨 Churn Prediction: YES - The employee is likely to leave.</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p style="color: green; font-weight: bold;">✅ Churn Prediction: NO - The employee is likely to stay.</p>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# --- SHAP Explainability ---
st.subheader("📊 SHAP Explainability (No Button Needed)")

try:
    # Load and preprocess full dataset
    df = pd.read_csv('HR_Analytics.csv').drop_duplicates()
    df_nums = df.select_dtypes(exclude='object')
    df_objs = pd.get_dummies(df.select_dtypes(include='object'), drop_first=True)
    df_full = pd.concat([df_nums, df_objs], axis=1)

    X = df_full.drop("left", axis=1)
    y = df_full["left"]

    # Train a new XGBoost model for SHAP
    xgb_model = XGBClassifier(random_state=42, eval_metric='logloss')
    xgb_model.fit(X, y)

    # SHAP explanation
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X)

    # SHAP summary plot
    st.subheader("🔍 SHAP Summary Plot")
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    shap.summary_plot(shap_values, X, show=False)
    st.pyplot(fig1)

    # SHAP bar plot
    st.subheader("📌 SHAP Feature Importance (Bar Plot)")
    shap_exp = shap.Explanation(values=shap_values, base_values=explainer.expected_value, data=X, feature_names=X.columns)
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    shap.plots.bar(shap_exp, show=False)
    st.pyplot(fig2)

except Exception as e:
    st.warning("⚠️ SHAP explanation failed. Please check the model and features.")
    st.text(f"Error: {str(e)}")
