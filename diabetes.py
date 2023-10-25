import streamlit as st
import pickle

model = pickle.load(open('prediksi_diabetes.sav', 'rb'))

st.title('Diabetes Prediction App')

st.write('Please enter the following details for the prediction')

age = st.number_input('Age', 0, 80, 25)

hypertension = st.selectbox('Hypertension', ['Yes', 'No'])
if hypertension == 'Yes':
  hypertension = 1
else:
  hypertension = 0

heart_disease = st.selectbox('Heart Disease', ['Yes', 'No'])
if heart_disease == 'Yes':
  heart_disease = 1
else:
  heart_disease = 0

smoking_history = st.selectbox('Smoking History', ['formerly smoked', 'never smoked', 'smokes', 'Unknown'])
if smoking_history == 'formerly smoked':
  smoking_history = 1
elif smoking_history == 'never smoked':
  smoking_history = 2
elif smoking_history == 'smokes':
  smoking_history = 3
else:
  smoking_history = 0

bmi = st.number_input('BMI', 10.0, 95.7, 20.0)

HbA1c_level = st.number_input('HbA1c Level', 3.5, 9.0, 4.0)

blood_glucose_level = st.number_input('Blood Glucose Level', 80, 300, 100)

#make a button to start the prediction
if st.button('Predict'):
  prediction = model.predict([[age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level]])
  if prediction == 0:
    st.write('You are not likely to have diabetes')
  else:
    st.write('You are likely to have diabetes')