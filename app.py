import streamlit as st
import pickle
import pandas as pd
import shap
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

# Page config & CSS tweaks
st.set_page_config(page_title="Flight Risk Employee Prediction", layout="wide")
st.markdown("""
<style>
.prediction-box { background-color: #D6EAF8; padding: 20px; border-radius: 12px;
  border: 1px solid #2980B9; box-shadow: 2px 2px 6px rgba(0,0,0,0.1); margin-bottom: 20px;}
.prediction-title { font-size: 24px; font-weight: bold; color: #1B4F72; margin-bottom: 10px; }
</style>""", unsafe_allow_html=True)

# Sidebar inputs...
# (same as before, omitted for brevity)

# Load and preprocess user input
# ... (same)

# Load model and scaler
# ... (same)

# Prediction section
# ... (same)

# --- FULL SHAP EXPLANATIONS ---
st.subheader("🔬 Comprehensive SHAP Analysis (Auto-generated)")

# Load and preprocess full dataset
df = pd.read_csv('HR_Analytics.csv').drop_duplicates()
df_nums = df.select_dtypes(exclude='object')
df_objs = pd.get_dummies(df.select_dtypes(include='object'), drop_first=True)
df_full = pd.concat([df_nums, df_objs], axis=1)
X = df_full.drop("left", axis=1)
y = df_full["left"]

# Train SHAP-compatible model
xgb_model = XGBClassifier(random_state=42, eval_metric='logloss')
xgb_model.fit(X, y)
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X)

# 1️⃣ Summary Plot
st.subheader("1️⃣ SHAP Summary Plot")
fig1, ax1 = plt.subplots(figsize=(7, 5))
shap.summary_plot(shap_values, X, show=False)
st.pyplot(fig1)
st.markdown("> **Interpretation:** Each dot represents an employee. Features are ranked by impact: red = high feature value, blue = low. Spread shows how strongly the feature drives churn positively or negatively.")

# 2️⃣ Bar Plot of Mean |SHAP|
st.subheader("2️⃣ SHAP Feature Importance (Bar Plot)")
fig2, ax2 = plt.subplots(figsize=(7, 5))
shap_exp = shap.Explanation(values=shap_values, base_values=explainer.expected_value, data=X, feature_names=X.columns)
shap.plots.bar(shap_exp, show=False)
st.pyplot(fig2)
st.markdown("> **Interpretation:** This bar chart shows average absolute impact of each feature on churn prediction, ranking them clearly by importance.")

# 3️⃣ Dependence Plots for top 3 features
top3 = pd.Series(abs(shap_values).mean(axis=0), index=X.columns).sort_values(ascending=False).head(3)
for feature in top3.index:
    st.subheader(f"3️⃣ SHAP Dependence Plot: {feature}")
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    shap.dependence_plot(feature, shap_values, X, ax=ax3, show=False)
    st.pyplot(fig3)
    st.markdown(f"> **Interpretation:** Shows how changes in **{feature}** affect churn risk, and how interaction with another feature adds nuance.")

# 4️⃣ Force Plot for the selected input
st.subheader("4️⃣ Global SHAP Force Plot for Prediction Set")
# Aggregate mean feature values for illustrative force
mean_feat = X.mean().values.reshape(1, -1)
shap_values_mean = explainer.shap_values(mean_feat)
fig4 = shap.force_plot(explainer.expected_value, shap_values_mean, X.columns, matplotlib=True)
st.pyplot(fig4)
st.markdown("> **Interpretation:** Visualizes how each feature pushes the prediction from base value; red pushes toward churn, blue away.")

# 5️⃣ Interactive force for user input
st.subheader("5️⃣ SHAP Force Plot for Your Input")
shap_values_input = explainer.shap_values(scaled_input) if hasattr(scaler_churn, 'transform') else explainer.shap_values(input_data)
fig5 = shap.force_plot(explainer.expected_value, shap_values_input[0], feature_names=X.columns, matplotlib=True)
st.pyplot(fig5)
st.markdown("> **Interpretation:** Highlights feature contributions for your specific scenario, showing what's tipping the prediction in either direction.")

