from loan_predictor import LoanPredictor
import pandas as pd

def predict_loan_status(data):
    predictor = LoanPredictor()
    predictor.load_model()
    
    if isinstance(data, dict):
        data = pd.DataFrame([data])
    
    data = predictor.clean_gender(data)
    
    data = predictor.handle_outliers(data)
    
    predictions = predictor.predict(data)
    probabilities = predictor.predict_proba(data)
    
    return predictions, probabilities

if __name__ == "__main__":
    sample_data = {
        'person_age': 22,
        'person_gender': 'female',
        'person_education': 'Master',
        'person_income': 71948.0,
        'person_emp_exp': 0,
        'person_home_ownership': 'RENT',
        'loan_amnt': 35000.0,
        'loan_intent': 'PERSONAL',
        'loan_int_rate': 16.02,
        'loan_percent_income': 0.49,
        'cb_person_cred_hist_length': 3.0,
        'credit_score': 561,
        'previous_loan_defaults_on_file': 'No'
    }
    
    predictions, probabilities = predict_loan_status(sample_data)
    
    approval_probability = probabilities[0][1] * 100
    rejection_probability = probabilities[0][0] * 100
    
    print(f"Prediction: {predictions[0]}")
    print(f"Probability of Approval: {approval_probability:.2f}%")
    print(f"Probability of Rejection: {rejection_probability:.2f}%")