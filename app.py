import streamlit as st
import pickle
import pandas as pd
from PIL import Image
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

st.set_page_config(page_title="Flight Risk Employee Prediction", layout="wide")

# --- CSS styles ---
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
        background-color: #ADD8E6;  /* Light blue background */
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

# --- Header ---
st.markdown('<div class="title">🧠 Flight Risk Employee Prediction</div>', unsafe_allow_html=True)

# --- Footer Image ---
with st.expander("ℹ️ Model Info", expanded=True):
    st.markdown("""
        <div class="info-box">
        <b>Recall and Accuracy Scores:</b><br>
        • <b>XGBoost:</b> Recall (1) = 0.91, Accuracy = 0.98
        </div>
    """, unsafe_allow_html=True)

# --- Sidebar ---
st.subheader("🎛️ Intelligent Employee Retention System for Churn Prediction")
satisfaction_level  = st.sidebar.slider("Satisfaction Level" , 0, 10)
last_evaluation     = st.sidebar.slider("Last Evaluation"    , 0, 10)
number_project      = st.sidebar.slider("Number Project"      , 2, 7)
average_montly_hours= st.sidebar.slider("Average Montly Hours" , 0, 500, step=8)
time_spend_company  = st.sidebar.slider("Time Spend Company"  , 1, 10)
Work_accident       = st.sidebar.slider("Work Accident"        , 0,1)
promotion_last_5years= st.sidebar.slider("Promotion Last 5 Years", 0,1)
Departments = st.sidebar.radio("Departments", ("sales","IT","RandD","Departments_hr","mng","support","technical"))
salary      = st.sidebar.radio("Salary", ("low","medium","high"))
model_name= st.selectbox("Select your model:", ("XGB Model", ))

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

# Show user input dataframe
st.subheader("📋 Selected Configuration")
st.dataframe(predictions)

# --- When Predict button pressed ---
if st.button("Predict"):

    # Preprocessing
    predictions['salary'] = predictions['salary'].map({'low': 0, 'medium': 1, 'high': 2}).astype(int)
    columns = ['satisfaction_level', 'last_evaluation', 'number_project',
               'average_montly_hours', 'time_spend_company', 'Work_accident',
               'promotion_last_5years', 'salary', 'Departments_IT',
               'Departments_RandD', 'Departments_accounting', 'Departments_hr',
               'Departments_management', 'Departments_marketing',
               'Departments_product_mng', 'Departments_sales', 'Departments_support',
               'Departments_technical']

    predictions = pd.get_dummies(predictions).reindex(columns=columns, fill_value=0)

    # Scale
    scaler_churn = pickle.load(open("scaler_churn", "rb"))
    scaled_predictions = scaler_churn.transform(predictions)

    st.subheader("⚙️ Scaled Features")
    st.dataframe(pd.DataFrame(scaled_predictions, columns=columns))

    # Load model
    if model_name == "XGB Model":
        model = pickle.load(open("XGB_model", "rb"))

    # Prediction result box with blue background
    st.markdown('<div class="prediction-box"><div class="prediction-title">🎯 Prediction Result</div>', unsafe_allow_html=True)

    prediction = model.predict(scaled_predictions)
    if int(prediction) == 1:
        st.markdown('<p style="color: red; font-weight: bold;">🚨 Churn Prediction: YES - The employee is likely to leave.</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p style="color: green; font-weight: bold;">✅ Churn Prediction: NO - The employee is likely to stay.</p>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Load dataset for explainer (ensure this matches model training data)
    df = pd.read_csv('HR_Analytics.csv').drop_duplicates()
    df_nums = df.select_dtypes(exclude='object')
    df_objs = df.select_dtypes(include='object')
    df_objs = pd.get_dummies(df_objs, drop_first=True)
    df = pd.concat([df_nums, df_objs], axis=1)

    X = df.drop("left", axis=1)
    y = df.left
    X = X.reindex(columns=columns, fill_value=0)

    # SHAP Explainer
    st.subheader("📊 SHAP Explanations")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # 1️⃣ SHAP Summary Plot (Feature Impact)
    fig1, ax1 = plt.subplots(figsize=(8,5))
    shap.summary_plot(shap_values, X, show=False)
    st.pyplot(fig1)
    st.markdown("**Interpretation:** Red indicates features pushing towards churn; blue indicates features supporting retention. The farther from the center, the stronger the influence.")

    # 2️⃣ SHAP Bar Plot (Mean Absolute SHAP Values)
    shap_exp = shap.Explanation(
        values=shap_values,
        base_values=explainer.expected_value,
        data=X,
        feature_names=X.columns)
    fig2, ax2 = plt.subplots(figsize=(8,5))
    shap.plots.bar(shap_exp, show=False)
    st.pyplot(fig2)
    st.markdown("**Interpretation:** This plot ranks features by their average impact across the dataset. The top features are most influential for churn predictions.")

    # 3️⃣ SHAP Beeswarm Plot
    fig3, ax3 = plt.subplots(figsize=(8,5))
    shap.plots.beeswarm(shap_exp, show=False)
    st.pyplot(fig3)
    st.markdown("**Interpretation:** Each dot represents a prediction. Clusters reveal patterns in feature contributions, colored by feature values.")

    # 4️⃣ Global SHAP Force Plot for Average Profile
    mean_shap_values = shap_values.mean(axis=0)
    fig4, ax4 = plt.subplots(figsize=(10,3))
    shap.force_plot(explainer.expected_value, mean_shap_values, X.iloc[0], matplotlib=True, show=False)
    st.pyplot(fig4)
    st.markdown("**Interpretation:** Red features push towards churn; blue push towards retention for an average employee profile.")

    # 5️⃣ SHAP Force Plot for Your Input
    fig5, ax5 = plt.subplots(figsize=(10,3))
    shap.force_plot(explainer.expected_value, shap_values[0], scaled_predictions[0], matplotlib=True, show=False)
    st.pyplot(fig5)
    st.markdown("**Interpretation:** Shows how your input features push the prediction toward churn or retention.")

    # 6️⃣ SHAP Waterfall Plot for Your Input
    fig6, ax6 = plt.subplots(figsize=(10,5))
    shap.plots.waterfall(shap.Explanation(values=shap_values[0],
                                         base_values=explainer.expected_value,
                                         data=X.iloc[0],
                                         feature_names=X.columns), show=False)
    st.pyplot(fig6)
    st.markdown("**Interpretation:** Breaks down the prediction for your input feature by feature, showing the cumulative effect.")

    # SHAP Dependence plots for top 3 features
    top_features = X.columns[:3]
    for feature in top_features:
        st.subheader(f"🔍 SHAP Dependence Plot: {feature}")
        fig_dp, ax_dp = plt.subplots(figsize=(8,4))
        shap.dependence_plot(feature, shap_values, X, show=False, ax=ax_dp)
        st.pyplot(fig_dp)
        st.markdown(f"**Interpretation:** Shows how the value of **{feature}** affects its contribution to churn prediction across the dataset.")

else:
    st.info("Click the **Predict** button to run prediction and see SHAP explanations.")
