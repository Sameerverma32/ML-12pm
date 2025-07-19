import pandas as pd 
import numpy as np 
import streamlit as st
import joblib

with open("models\diabetes.pkl", "rb") as f:
    model = joblib.load(f)  


# Page config
st.set_page_config(page_title='Diabetes Predictor', layout='wide')

st.markdown("""
    <style>
    .stApp {
        background-color: #1a1a2e;
        color: #ffffff;
    }

    .custom-box {
        background-color: #16213e;
        border: 2px solid #0f3460;
        padding: 30px;
        border-radius: 15px;
        margin-bottom: 25px;
        box-shadow: 3px 3px 10px rgba(0, 0, 0, 0.5);
    }

    .stButton > button {
        background-color: #e94560;
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 8px;
        padding: 0.6em 1.2em;
        transition: background-color 0.3s ease;
    }

    .stButton > button:hover {
        background-color: #ff5f77;
        color: black;
    }

    .stSlider > div {
        color: white;
    }


    </style>
""", unsafe_allow_html=True)

# Title
st.title('🧬 Diabetes Prediction App')
st.markdown("This app predicts whether a person has diabetes based on their medical attributes.")

# Sidebar input
st.sidebar.header('📝 Input Patient Data')
pregnancies = st.sidebar.slider('Pregnancies', 0, 20, 1)
glucose = st.sidebar.slider('Glucose Level', 0, 200, 100)
blood_pressure = st.sidebar.slider('Blood Pressure', 0, 150, 70)    
skin_thickness = st.sidebar.slider('Skin Thickness', 0, 100, 20)
insulin = st.sidebar.slider('Insulin Level', 0, 500, 100)
bmi = st.sidebar.slider('BMI', 0.0, 50.0, 25.0, step=0.1)
dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 2.5, 0.5, step=0.01)
age = st.sidebar.slider('Age', 0, 120, 30)

col1, col2 = st.columns([1, 2])

with col1 :
    #st.markdown('<div class="custom-box">', unsafe_allow_html=True)
    st.subheader('📝Input Data Summary')
    st.write({
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "Blood Pressure": blood_pressure,
        "Skin Thickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DPF": dpf,
        "Age": age
    })

    if st.button('🔍 Predict'):
        features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
        predicted_class = model.predict(features)[0]
    
        # Show prediction
        if predicted_class == 1:
            st.success('🩺 Predicted: Diabetes Positive')
        else:
            st.success('🩺 Predicted: Diabetes Negative')
        st.info("Prediction based on the trained classification model.")
    st.markdown('</div>', unsafe_allow_html=True)

    try:
        import plotly.express as px
        probabilities = model.predict_proba(features)[0]
        fig = px.pie(
            values=probabilities,
            names=['Negative', 'Positive'],
            title='Diabetes Risk Distribution',
            hole=0.4,
            color_discrete_sequence=["#00cc96", "#EF553B"]
        )
        fig.update_layout(width=350, height=350)  # Square shape
        st.plotly_chart(fig, use_container_width=False)
    except:
        st.warning("Pie chart not available. Provide the inputs from the sidebar .")

with col2:
    #st.markdown('<div class="custom-box">', unsafe_allow_html=True)
    st.subheader("📈 Feature Importance")
    try:
        feature_importance = model.feature_importances_
        feature_names = ['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness', 'Insulin', 'BMI', 'DPF', 'Age']
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
        st.bar_chart(importance_df.set_index('Feature'))
        st.dataframe(importance_df.sort_values(by='Importance', ascending=False), use_container_width=True)
    except:
        st.warning("Feature importance not available for this model.")
    st.markdown('</div>', unsafe_allow_html=True)
    