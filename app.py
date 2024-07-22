import streamlit as st
import numpy as np
import joblib

# Load the trained model
clf = joblib.load('financial_inclusion_model.pkl')

# Define the Streamlit app
def main():
    st.title("Financial Inclusion Prediction")

    # Input fields for features
    feature_1 = st.text_input('Feature 1')
    feature_2 = st.text_input('Feature 2')
    # Add more input fields as necessary

    if st.button('Predict'):
        # Process inputs and make prediction
        input_data = np.array([[feature_1, feature_2]])  # Adjust according to your features
        prediction = clf.predict(input_data)
        
        st.write("Prediction:", prediction[0])

# Run the app
if __name__ == '__main__':
    main()
