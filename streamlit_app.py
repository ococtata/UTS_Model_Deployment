from loan_predictor import LoanPredictor 
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

@st.cache_resource
def load_predictor():
    predictor = LoanPredictor()
    predictor.load_model()
    return predictor

predictor = load_predictor()

images = {
    "Approved": "https://www.shutterstock.com/image-vector/approved-green-square-rubber-stamp-260nw-1927615946.jpg",
    "Rejected": "https://media.istockphoto.com/id/949546382/vector/rejected-ink-stamp.jpg?s=612x612&w=0&k=20&c=S8MRCa7JMK7cSNvQwflyDLyMzXAZ3ng3vRw7rVP9eNU="
}

def display_prediction_results(predictions, probabilities):
    col1, col2 = st.columns(2)
    
    with col1:
        status = predictions[0]
        
        if status == "Approved" or status == 1:
            st.success(f"Loan Approval Status: Approved")
            image_url = images["Approved"]
        else:
            st.error(f"Loan Approval Status: Rejected")
            image_url = images["Rejected"]
        
        st.write(f"Probability of Approval: {probabilities[0][1]:.2%}")
        st.write(f"Probability of Rejection: {probabilities[0][0]:.2%}")
        st.image(image_url, use_container_width=True)
    
    with col2:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.pie(
            [probabilities[0][1], probabilities[0][0]], 
            labels=['Approved', 'Rejected'],
            autopct='%1.1f%%',
            startangle=90,
            colors=['#4CAF50', '#F44336']
        )
        ax.axis('equal')
        st.pyplot(fig)

st.title("UTS Model Deployment - Loan Approval Prediction")
st.write("""This application predicts if a loan application will be approved or rejected.
\n\tMade by Octavius Sandriago - 2702221135""")

tab1, tab2, tab3 = st.tabs(["Prediction Form", "Test Cases", "Information"])

with tab1:
    st.header("Applicant Information")
    col1, col2 = st.columns(2)

    with col1:
        person_age = st.number_input("Usia", min_value=1, max_value=100, value=18)
        person_gender = st.selectbox("Gender", ["male", "female"])
        person_education = st.selectbox("Tingkat pendidikan tertinggi", ["High School", "Associate", "Bachelor", "Master", "Doctorate"])
        person_income = st.number_input("Pendapatan tahunan ($)", min_value=0)
        person_emp_exp = st.number_input("Tahun pengalaman bekerja", min_value=0)
        person_home_ownership = st.selectbox("Status Kepemilikan Tempat Huni", ["RENT", "MORTGAGE", "OWN", "OTHER"], index=0)

    with col2:
        loan_amnt = st.number_input("Jumlah pinjaman yang diminta ($)", min_value=1000, max_value=100000, value=10000)
        loan_intent = st.selectbox("Tujuan Dari Pinjaman", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"], index=0)
        loan_int_rate = st.number_input("Suku Bunga Pinjaman (%)", min_value=1.0, max_value=30.0, step=0.1)
        loan_percent_income = st.number_input("Persentase Pinjaman dari Pendapatan", min_value=0.01, max_value=1.0, step=0.01)
        cb_person_cred_hist_length = st.number_input("Durasi Kredit (tahun)", min_value=0)
        credit_score = st.slider("Skor Kredit", min_value=300, max_value=850, step=1)

    previous_loan_defaults_on_file = st.selectbox("Tunggakan Pinjaman Sebelumnya", ["Yes", "No"])

    if st.button("Predict Loan Approval"):
        input_data = {
            'person_age': person_age,
            'person_gender': person_gender,
            'person_education': person_education,
            'person_income': person_income,
            'person_emp_exp': person_emp_exp,
            'person_home_ownership': person_home_ownership,
            'loan_amnt': loan_amnt,
            'loan_intent': loan_intent,
            'loan_int_rate': loan_int_rate,
            'loan_percent_income': loan_percent_income,
            'cb_person_cred_hist_length': cb_person_cred_hist_length,
            'credit_score': credit_score,
            'previous_loan_defaults_on_file': previous_loan_defaults_on_file
        }
        predictions = predictor.predict(input_data)
        probabilities = predictor.predict_proba(input_data)
        st.header("Prediction Results")
        display_prediction_results(predictions, probabilities)

with tab2:
    st.header("Test Cases")
    st.write("Here are two pre-defined test cases to demonstrate the model's predictions.")

    st.subheader("Test Case 1: Approved")
    test_case1 = {
        'person_age':22,
        'person_gender': 'female',
        'person_education': 'Master',
        'person_income': 71948,
        'person_emp_exp': 0,
        'person_home_ownership': 'RENT',
        'loan_amnt': 35000,
        'loan_intent': 'PERSONAL',
        'loan_int_rate': 16.02,
        'loan_percent_income': 0.49,
        'cb_person_cred_hist_length': 3,
        'credit_score': 561,
        'previous_loan_defaults_on_file': 'No'
    }

    st.json(test_case1)

    if st.button("Run Test Case 1"):
        predictions1 = predictor.predict(test_case1)
        probabilities1 = predictor.predict_proba(test_case1)
        display_prediction_results(predictions1, probabilities1)

    st.subheader("Test Case 2: Rejected")
    test_case2 = { 
        'person_age': 21,
        'person_gender': 'female',
        'person_education': 'High School',
        'person_income': 12282,
        'person_emp_exp': 0,
        'person_home_ownership': 'OWN',
        'loan_amnt': 1000,
        'loan_intent': 'EDUCATION',
        'loan_int_rate': 11.14,
        'loan_percent_income': 0.08,
        'cb_person_cred_hist_length': 2,
        'credit_score': 504,
        'previous_loan_defaults_on_file': 'Yes'
    }

    st.json(test_case2)

    if st.button("Run Test Case 2"):
        predictions2 = predictor.predict(test_case2)
        probabilities2 = predictor.predict_proba(test_case2)
        display_prediction_results(predictions2, probabilities2)

with tab3:
    st.header("Information")
    st.write("""    
    ### Key Features Used:
    
    - person_age = Usia dari orang tersebut
    - person_gender = Gender dari orang tersebut
    - person_education = Tingkat pendidikan tertinggi
    - person_income = Pendapatan tahunan
    - person_emp_exp = Tahun pengalaman bekerja
    - person_home_ownership = Status kepemilikan tempat huni
    - loan_amnt = Jumlah pinjaman yang diminta
    - loan_intent = Tujuan dari pinjaman
    - loan_int_rate = Suku bunga pinjaman
    - loan_percent_income = Jumlah pinjaman sebagai persentase dari pendapatan tahunan
    - cb_person_cred_hist_length = Durasi kredit dalam tahun
    - credit_score = Skor kredit dari orang tersebut
    - previous_loan_defaults_on_file = Indikator tunggakan pinjaman sebelumnya
    - loan_status (target variable) = Persetujuan pinjaman; 1: diterima dan 0: ditolak
    """)