import streamlit as st
import pickle
import pandas as pd
import shap
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from PIL import Image

st.set_page_config(page_title="Flight Risk Employee Prediction", layout="wide")

st.markdown("""
<style>
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
    color: #1B4F72;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="prediction-box"><div class="prediction-title">🧠 Flight Risk Employee Prediction</div>', unsafe_allow_html=True)

# Sidebar inputs
st.sidebar.subheader("🎛️ Intelligent Employee Retention System for Churn Prediction")
satisfaction_level = st.sidebar.slider("Satisfaction Level", 0, 10)
last_evaluation = st.sidebar.slider("Last Evaluation", 0, 10)
number_project = st.sidebar.slider("Number Project", 2, 7)
average_montly_hours = st.sidebar.slider("Average Montly Hours", 0, 500, step=8)
time_spend_company = st.sidebar.slider("Time Spend Company", 1, 10)
Work_accident = st.sidebar.slider("Work Accident", 0, 1)
promotion_last_5years = st.sidebar.slider("promotion_last_5years", 0, 1)
Departments = st.sidebar.radio("Departments", ("sales", "IT", "RandD", "Departments_hr", "mng", "support", "technical"))
salary = st.sidebar.radio("Salary", ("low", "medium", "high"))
model_name = st.selectbox("Select your model:", ("XGB Model",))

# Input dataframe
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

# Preprocessing
input_data['salary'] = input_data['salary'].map({'low': 0, 'medium': 1, 'high': 2}).astype(int)
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

# Load model
model = pickle.load(open("XGB_model", "rb"))

# Predict button
if st.button("Predict Churn"):
    prediction = model.predict(scaled_input)
    if int(prediction) == 1:
        st.markdown('<p style="color: red; font-weight: bold;">🚨 Churn Prediction: YES - The employee is likely to leave.</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p style="color: green; font-weight: bold;">✅ Churn Prediction: NO - The employee is likely to stay.</p>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Load dataset
df = pd.read_csv('HR_Analytics.csv').drop_duplicates()
df_nums = df.select_dtypes(exclude='object')
df_objs = pd.get_dummies(df.select_dtypes(include='object'), drop_first=True)
df_full = pd.concat([df_nums, df_objs], axis=1)
X = df_full.drop("left", axis=1)
y = df_full["left"]

# SHAP explanation
st.subheader("🔬 Comprehensive SHAP Analysis (Auto-generated)")
xgb_model = XGBClassifier(random_state=42, eval_metric='logloss')
xgb_model.fit(X, y)
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X)

# 1️⃣ Summary Plot
st.subheader("1️⃣ SHAP Summary Plot")
fig1, ax1 = plt.subplots(figsize=(7, 5))
shap.summary_plot(shap_values, X, show=False)
st.pyplot(fig1)
st.markdown("> **Interpretation:** Shows feature importance and their direction (positive or negative) on churn.")

# 2️⃣ Bar Plot
st.subheader("2️⃣ SHAP Feature Importance (Bar Plot)")
shap_exp = shap.Explanation(values=shap_values, base_values=explainer.expected_value, data=X, feature_names=X.columns)
fig2, ax2 = plt.subplots(figsize=(7, 5))
shap.plots.bar(shap_exp, show=False)
st.pyplot(fig2)
st.markdown("> **Interpretation:** Average absolute contribution of each feature to predictions.")

# 3️⃣ Dependence plots
top3 = pd.Series(abs(shap_values).mean(axis=0), index=X.columns).sort_values(ascending=False).head(3)
for feature in top3.index:
    st.subheader(f"3️⃣ SHAP Dependence Plot: {feature}")
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    shap.dependence_plot(feature, shap_values, X, ax=ax3, show=False)
    st.pyplot(fig3)
    st.markdown(f"> **Interpretation:** How {feature} values affect churn risk, possibly with interaction effects.")

# 4️⃣ Force Plot for average
st.subheader("4️⃣ Global SHAP Force Plot for Average Profile")
mean_feat = X.mean().values.reshape(1, -1)
shap_values_mean = explainer.shap_values(mean_feat)
fig4 = shap.force_plot(explainer.expected_value, shap_values_mean[0], feature_names=X.columns, matplotlib=True)
st.pyplot(fig4)
st.markdown("> **Interpretation:** Red features push towards churn; blue push to retain.")

# 5️⃣ Force Plot for user input
import streamlit.components.v1 as components
st.subheader("5️⃣ SHAP Force Plot for Your Input")
# Get SHAP values for the input (make sure shap_input is the correct output shape)
shap_input = explainer.shap_values(scaled_input)

# Create interactive force plot (matplotlib=False)
force_plot_html = shap.force_plot(
    explainer.expected_value, 
    shap_input[0],  # assuming shap_input[0] corresponds to your input
    scaled_input[0],  # pass the input features (scaled) for display
    feature_names=X.columns,
    matplotlib=False
)

# Embed the plot as HTML inside Streamlit
components.html(force_plot_html.html(), height=400)

st.markdown("> **Interpretation:** Breakdown of your input's impact on churn prediction.")

