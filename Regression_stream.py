import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the trained model
model = tf.keras.models.load_model('Regression_model.h5')

# Load the encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler_reg.pkl', 'rb') as file:
    scaler_reg = pickle.load(file)



## streamlit app
st.title('Estimated Salary Prediction')

# User input
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
credit_score = st.number_input('Credit Score')

balance = 0
tenure = 5
num_of_products = 1
has_cr_card = 1
is_active_member = 1
geography = 'France'


# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
})



# One-hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler_reg.transform(input_data)



# Predict churn
prediction = model.predict(input_data_scaled)
prediction_salary = prediction[0][0]

#st.write(f'Prediction Estimated Salary: {prediction_salary:.2f}')

st.subheader("🔎 Prediction Result")
st.write("### User Info")
st.write(f"**Gender:** {gender}")
st.write(f"**Age:** {age}")
st.write(f"**Credit Score:** {credit_score}")

st.success(f"💰 Prediction Estimated Salary: **{prediction_salary:,.2f}**")











#if prediction_salary > 50000:
    #st.write('The customer salary is good.')
#else:
    #st.write('The customer salary is avg.')



