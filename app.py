import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open('financial_inclusion_model_1.pkl', 'rb') as file:
    model = pickle.load(file)

# Define the input fields
st.title('Financial Inclusion Prediction')

age = st.number_input('Age', min_value=0, max_value=120, value=30)
country = st.selectbox('Country', ['Tanzania', 'Kenya', 'Uganda', 'Rwanda'])

# Create a button to make the prediction
if st.button('Predict'):
    # Create a DataFrame with the user inputs
    input_data = pd.DataFrame({
        'age_of_respondent': [age],  # Use the correct column name
        'country': [country]
    })
    
    # Preprocess the input data as needed
    input_data['age_of_respondent'] = input_data['age_of_respondent'].fillna(input_data['age_of_respondent'].mean())  # Example for filling missing values
    #input_data = pd.get_dummies(input_data)  # Example for encoding categorical variables
    
    # Make a prediction
    prediction = model.predict(input_data)
    
    # Display the prediction result
    result = 'Has a bank account' if prediction[0] == 1 else 'Does not have a bank account'
    st.write(f'Prediction: {result}')
