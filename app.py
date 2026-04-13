import streamlit as st
import pandas as pd
import joblib

# --- 1. मॉडल लोड करें ---
# यह सबसे सुरक्षित तरीका है। यह 'salary_model.pkl' फाइल को सीधे पढ़ेगा।
try:
    model = joblib.load('salary_model.pkl')
except Exception as e:
    st.error(f"मॉडल फाइल नहीं मिली: {e}. कृपया सुनिश्चित करें कि आपने नोटबुक में 'joblib.dump' वाली सेल रन की है।")

def run_frontend():
    st.set_page_config(page_title="AI Salary Predictor", layout="centered")
    st.title("💰 AI Salary Prediction Tool")
    st.write("Enter your professional details below to estimate your market value.")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=80, value=25)
        gender = st.selectbox("Gender", ["Male", "Female"])
        experience = st.slider("Years of Experience", 0.0, 40.0, 5.0)

    with col2:
        edu_options = ["High School", "Bachelor's", "Master's", "PhD"]
        education = st.selectbox("Education Level", edu_options)
        job_title = st.text_input("Job Title", value="Software Engineer")

    if st.button("Predict My Salary"):
        # ध्यान दें: कॉलम का क्रम वही होना चाहिए जो ट्रेनिंग के समय 'X' में था
        # आपके ट्रेनिंग कोड के अनुसार क्रम है: ['Education Level', 'Job Title', 'Gender', 'Age', 'Years of Experience']
        user_input = pd.DataFrame([[education, job_title, gender, age, experience]], 
                                  columns=['Education Level', 'Job Title', 'Gender', 'Age', 'Years of Experience'])
        try:
            prediction = model.predict(user_input)[0]
            st.success(f"### Estimated Annual Salary: ${prediction:,.2f}")
            st.info("💡 Note: This is an AI estimate based on historical Kaggle data.")
        except Exception as e:
            st.error(f"Prediction Error: {e}")

if __name__ == "__main__":
    run_frontend()
