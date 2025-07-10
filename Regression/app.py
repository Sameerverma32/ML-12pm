import streamlit as st
import joblib 
import numpy as np

#load the model
with open('models\simple_linear_regression.pkl', 'rb') as file:
    model = joblib.load(file)
    
    #title
st.set_page_config(page_title='Salary Predictor')
st.title('Salary Predictor App')
st.subheader('Predict your salary based on your experience')
st.markdown("Simply input your years of experience and get the predicted salary.")

#side bar
st.sidebar.header('Enter your details👇')
experience = st.sidebar.slider("Years of Experience", 
                               min_value=0.0, max_value=20.0, step=0.5)

#Buttons to predict
if st.sidebar.button("Predict Salary"):
    
  #predict salary
  salary = model.predict(np.array([[experience]]))[0]

  #Display result
  st.success(f"Predicted Salary: Rs.{salary:,.2f}")

 #Additional Info
  st.info("This prediction is based on a simple Linear Regression")

#footer 
st.markdown("-------")
st.markdown("Made with using Streamlit")