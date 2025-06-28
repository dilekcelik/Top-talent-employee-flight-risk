import streamlit as st
import pickle
import pandas as pd
from PIL import Image

st.set_page_config(page_title="Employee Churn Predictor", layout="wide")
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
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🔍 Intelligent Employee Retention System for Churn Prediction</div>', unsafe_allow_html=True)

# --- Footer Image ---

with st.expander("ℹ️ Model Info", expanded=True):
    st.markdown("""
        <div class="info-box">
        <b>Recall and Accuracy Scores:</b><br>
        • <b>XGBoost:</b> Recall (1) = 0.97, Accuracy = 0.99
        </div>
    """, unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.header("🎛️ Input Features")
satisfaction_level = st.sidebar.slider("Satisfaction Level", 0, 10, 5)
last_evaluation = st.sidebar.slider("Last Evaluation", 0, 10, 5)
number_project = st.sidebar.slider("Number of Projects", 2, 7, 4)
average_montly_hours = st.sidebar.slider("Average Monthly Hours", 0, 500, 180, step=8)
time_spend_company = st.sidebar.slider("Years at Company", 1, 10, 3)
Work_accident = st.sidebar.radio("Work Accident", [0, 1])
promotion_last_5years = st.sidebar.radio("Promotion in Last 5 Years", [0, 1])
Departments = st.sidebar.selectbox("Department", ("sales", "IT", "RandD", "Departments_hr", "mng", "support", "technical"))
salary = st.sidebar.radio("Salary", ("low", "medium", "high"))
model_name = st.sidebar.selectbox("Select Model", ("XGB Model"))

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

# --- Display User Input ---
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

# --- Scale ---
st.subheader("⚙️ Scaled Features")
scaler_churn = pickle.load(open("scaler_churn", "rb"))
scaled_predictions = scaler_churn.transform(predictions)
st.dataframe(pd.DataFrame(scaled_predictions, columns=columns))

# --- Model Selection ---
if model_name == "Gradient Boosting Model":
    model = pickle.load(open("GradientBoosting_model", "rb"))
if model_name == "Random Forest Model":
    model = pickle.load(open("RandomForest_model", "rb"))
if model_name == "XGB Model":
    model = pickle.load(open("XGB_model", "rb"))
    
# --- Prediction ---
st.subheader("🎯 Prediction Result")
if st.button("Predict Churn"):
    prediction = model.predict(scaled_predictions)
    if int(prediction) == 1:
        st.error("🚨 Churn Prediction: YES - The employee is likely to leave.")
    else:
        st.success("✅ Churn Prediction: NO - The employee is likely to stay.") 
# --- Images ---
# st.image(Image.open('models_performance.png'), width=800, caption="Model Performance")

#  st.image(Image.open('churn.png'), width=800, caption='Churn Insight Illustration')

# ------ Data ------
df = pd.read_csv('HR_Analytics.csv')
df=df.drop_duplicates() # drop dublicates
#One Hot Encoding
df_nums = df.select_dtypes(exclude='object')  # This will select numeric columns
df_objs = df.select_dtypes(include='object')  # This will select object (categorical) columns
df_objs = pd.get_dummies(df_objs, drop_first=True)  # drop_first=True to avoid multicollinearity
df = pd.concat([df_nums, df_objs], axis=1)
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)
df_scaled = pd.DataFrame(df_scaled, columns=df.columns)
from sklearn.model_selection import train_test_split
X = df.drop("left", axis=1)
y = df.left
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ------ Explainer------
st.subheader("📊 Explainer")
import shap
import matplotlib.pyplot as plt
# Train your model (assuming xgb_model is already trained)
# explainer = shap.Explainer(model)
# Get SHAP values
# shap_values = explainer.shap_values(X_test)
# Visualize
# shap.summary_plot(shap_values, X_test)
import shap
import matplotlib.pyplot as plt
try:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    st.subheader("🔍 SHAP Summary Plot (Feature Importance)")

    fig, ax = plt.subplots(figsize=(6, 4))  # smaller plot size
    shap.summary_plot(shap_values, X_test, show=False, plot_size=(6, 4))
    st.pyplot(fig)

   # SHAP Bar Plot - Convert to Explanation object first
    shap_exp = shap.Explanation(
        values=shap_values,
        base_values=explainer.expected_value,
        data=X_test,
        feature_names=X_test.columns)
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    shap.plots.bar(shap_exp, show=False)
    st.pyplot(fig2)
    
except Exception as e:
    st.warning("⚠️ SHAP explanation failed. Please check the model and features.")
    st.text(f"Error: {str(e)}")
