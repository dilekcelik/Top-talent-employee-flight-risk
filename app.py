import streamlit as st
import pickle
import pandas as pd
from PIL import Image
import shap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
import streamlit.components.v1 as components

st.set_page_config(page_title="Flight Risk Employee Prediction", layout="wide")

# --- Header ---
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
        color: #154360;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🧠 Flight Risk Employee Prediction</div>', unsafe_allow_html=True)

with st.expander("ℹ️ Model Info", expanded=True):
    st.markdown("""
        <div class="info-box">
        <b>Recall and Accuracy Scores:</b><br>
        • <b>XGBoost:</b> Recall (1) = 0.91, Accuracy = 0.98
        </div>
    """, unsafe_allow_html=True)

# --- Sidebar ---
st.subheader("🎛️ Intelligent Employee Retention System for Churn Prediction")
satisfaction_level  = st.sidebar.slider("Satisfaction Level", 0, 10, value=5)
last_evaluation     = st.sidebar.slider("Last Evaluation", 0, 10, value=5)
number_project      = st.sidebar.slider("Number Project", 2, 7, value=4)
average_montly_hours = st.sidebar.slider("Average Montly Hours", 0, 500, step=8, value=250)
time_spend_company  = st.sidebar.slider("Time Spend Company", 1, 10, value=5)
Work_accident       = st.sidebar.slider("Work Accident", 0, 1)
promotion_last_5years = st.sidebar.slider("Promotion Last 5 Years", 0, 1)
Departments         = st.sidebar.radio("Departments", ("sales", "IT", "RandD", "Departments_hr", "mng", "support", "technical"))
salary              = st.sidebar.radio("Salary", ("low", "medium", "high"))
model_name          = st.selectbox("Select your model:", ("XGB Model"))

# --- DataFrame Creation ---
predictions = pd.DataFrame([{
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
st.dataframe(predictions)

# --- Preprocessing ---
predictions['salary'] = predictions['salary'].map({'low': 0, 'medium': 1, 'high': 2}).astype(int)
columns = ['satisfaction_level', 'last_evaluation', 'number_project',
           'average_montly_hours', 'time_spend_company', 'Work_accident',
           'promotion_last_5years', 'salary', 'Departments_IT',
           'Departments_RandD', 'Departments_accounting', 'Departments_hr',
           'Departments_management', 'Departments_marketing',
           'Departments_product_mng', 'Departments_sales', 'Departments_support',
           'Departments_technical']
predictions = pd.get_dummies(predictions).reindex(columns=columns, fill_value=0)

st.subheader("⚙️ Scaled Features")
scaler_churn = pickle.load(open("scaler_churn", "rb"))
scaled_predictions = scaler_churn.transform(predictions)
st.dataframe(pd.DataFrame(scaled_predictions, columns=columns))

# --- Model Load ---
model = pickle.load(open("XGB_model", "rb"))

st.markdown('<div class="prediction-box"><div class="prediction-title">🎯 Prediction Result</div>', unsafe_allow_html=True)
if st.button("Predict Churn"):
    prediction = model.predict(scaled_predictions)
    if int(prediction) == 1:
        st.markdown('<p style="color: red; font-weight: bold;">🚨 Churn Prediction: YES - The employee is likely to leave.</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p style="color: green; font-weight: bold;">✅ Churn Prediction: NO - The employee is likely to stay.</p>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# --- Load and Prepare Data for SHAP ---
df = pd.read_csv('HR_Analytics.csv').drop_duplicates()
df_nums = df.select_dtypes(exclude='object')
df_objs = pd.get_dummies(df.select_dtypes(include='object'), drop_first=True)
df = pd.concat([df_nums, df_objs], axis=1)
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
X = df.drop("left", axis=1)
y = df.left
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
xgb_model = XGBClassifier(random_state=42, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

# --- SHAP Explainer ---
st.subheader("📊 SHAP Explanations")
explainer = shap.Explainer(xgb_model)
shap_values = explainer(X_test)

# 1️⃣ SHAP Summary Plot
st.markdown("### 🔍 SHAP Summary Plot (Feature Impact)")
fig1, ax1 = plt.subplots()
shap.summary_plot(shap_values, X_test, show=False)
st.pyplot(fig1)
st.info("""Red indicates features pushing towards churn; blue indicates features supporting retention. The farther from the center, the stronger the influence.""")

# 2️⃣ SHAP Bar Plot
st.markdown("### 📊 SHAP Bar Plot (Mean Absolute SHAP Values)")
fig2, ax2 = plt.subplots()
shap.plots.bar(shap_values, show=False)
st.pyplot(fig2)
st.info("""This plot ranks features by their average impact across the dataset. The top features are most influential for churn predictions.""")

# 3️⃣ SHAP Beeswarm Plot
st.markdown("### 🐝 SHAP Beeswarm Plot")
fig3, ax3 = plt.subplots()
shap.plots.beeswarm(shap_values, show=False)
st.pyplot(fig3)
st.info("""Each dot represents a prediction. Clusters reveal patterns in feature contributions, colored by feature values.""")

# 4️⃣ Global SHAP Force Plot
st.markdown("### 🌐 Global SHAP Force Plot for Average Profile")
avg_exp = shap.Explanation(values=shap_values.values.mean(0), base_values=explainer.expected_value, data=X_test.iloc[0], feature_names=X_test.columns.tolist())
components.html(shap.plots.force(avg_exp, matplotlib=False), height=200)
st.info("Red features push towards churn; blue push to retain. This is an aggregate view of the typical employee profile.")

# 5️⃣ Individual SHAP Force Plot
st.markdown("### 👤 SHAP Force Plot for Your Input")
custom_exp = explainer(pd.DataFrame(scaled_predictions, columns=X.columns))
components.html(shap.plots.force(custom_exp[0], matplotlib=False), height=200)
st.info("This visual shows how your input features contribute to the churn prediction.")
