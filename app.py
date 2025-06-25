!pip install scikit-learn
import streamlit as st
import pickle
import pandas as pd
import sklearn

st.title('Intelligent Employee Retention System for Churn Prediction by Dr Dilek Celik')
st.info("The prediction is based on Gradient Boosting Model with recall(1): 0.96 and accuracy:0.99 /// Random Forest Model with recall(1): 0.97 and accuracy:0.99 /// XGB with recall(1): 0.97 and accuracy:0.99")

from PIL import Image
st.image(Image.open('models_performance.png'), width=700)


st.sidebar.title('Select the Features')
satisfaction_level  =st.sidebar.slider("Satisfaction Level" , 0, 10)
last_evaluation     =st.sidebar.slider("Last Evaluation"    , 0, 10)
number_project      =st.sidebar.slider("Number Project"      , 2, 7)
average_montly_hours=st.sidebar.slider("Average Montly Hours" , 0, 500, step=8)
time_spend_company  =st.sidebar.slider("Time Spend Company"  , 1, 10)
Work_accident       =st.sidebar.slider("Work Accident"        , 0,1)
promotion_last_5years=st.sidebar.slider("promotion_last_5years", 0,1)
Departments =st.sidebar.radio("Departments", ("sales","IT","RandD","Departments_hr","mng","support","technical"))
salary      =st.sidebar.radio("Salary", ("low","medium","high"))
model_name=st.selectbox("Select your model:", ("Gradient Boosting Model","Random Forest Model","XGB Model" ))




predictions = {'satisfaction_level': satisfaction_level/10,
                'last_evaluation': last_evaluation/10,
                'number_project': number_project,
                'average_montly_hours':average_montly_hours, 
                'time_spend_company':time_spend_company, 
                'Work_accident':Work_accident,
                'promotion_last_5years':promotion_last_5years, 
                'salary':salary,
                'Departments':Departments}
predictions = pd.DataFrame([predictions])

st.header("The configuration is below")
st.table(predictions[['satisfaction_level', 'last_evaluation', 'number_project',
       'average_montly_hours', 'time_spend_company', 'Work_accident',
       'promotion_last_5years', 'salary', 'Departments']])


#LABEL ENCODING
predictions['salary'] = predictions['salary'].map({'low':0, 'medium':1, 'high':2}).astype(int)


#GET DUMMIES
columns=['satisfaction_level', 'last_evaluation', 'number_project',
       'average_montly_hours', 'time_spend_company', 'Work_accident',
       'promotion_last_5years', 'salary', 'Departments_IT',
       'Departments_RandD', 'Departments_accounting', 'Departments_hr',
       'Departments_management', 'Departments_marketing',
       'Departments_product_mng', 'Departments_sales', 'Departments_support',
       'Departments_technical']
predictions = pd.get_dummies(predictions).reindex(columns=columns, fill_value=0)



#SCALED
st.header("Scaled Data")
scaler_churn= pickle.load(open("scaler_churn", "rb"))
scaled_predictions= scaler_churn.transform(predictions)


#MODEL
if model_name=="Gradient Boosting Model":
    model = pickle.load(open("GradientBoosting_model","rb"))
    st.success("You selected {} model".format(model_name))
elif model_name=="Random Forest Model":
    model = pickle.load(open("RandomForest_model","rb"))
    st.success("You selected {} model".format(model_name))
else:
	model=pickle.load(open("XGB_model","rb"))
	st.success("You selected {} model".format(model_name))


#st.success(model.predict(predictions))


st.subheader("Press predict if configuration is okay")
if st.button("Predict"):
    prediction=model.predict(scaled_predictions)
    if int(prediction) ==1:
        st.error("Churn Prediction is {}. ".format("YES"))
    else:
        st.success("Churn Prediction is {}. ".format("NO"))

#importing image
from PIL import Image
img = Image.open('churn.png')
st.image(img, width=700, caption='Churn Prediction')
