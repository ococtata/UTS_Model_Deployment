import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder, OneHotEncoder

class LoanPredictor:
    def __init__(self):
        self.model = None
        self.min_max_scaler = None
        self.oh_encoder = None
        self.od_encoder = None
        self.numerical_cols = None
        self.categorical_cols = None
        self.non_hierarchal_cat = None
        self.hierarchal_col = None
    
    def load_model(self, model_path='best_model.pkl', preprocessing_path='preprocessing_objects.pkl'):
        with open(model_path, 'rb') as file:
            self.model = pickle.load(file)
        
        with open(preprocessing_path, 'rb') as file:
            preprocessing_objects = pickle.load(file)
            
        self.min_max_scaler = preprocessing_objects['min_max_scaler']
        self.oh_encoder = preprocessing_objects['oh_encoder']
        self.od_encoder = preprocessing_objects['od_encoder']
        self.numerical_cols = preprocessing_objects['numerical_cols']
        self.categorical_cols = preprocessing_objects['categorical_cols']
        self.non_hierarchal_cat = preprocessing_objects['non_hierarchal_cat']
        self.hierarchal_col = preprocessing_objects['hierarchal_col']
            
    def preprocess_data(self, data):
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        
        if 'person_income' in data.columns and data['person_income'].isna().any():
            data['person_income'] = data['person_income'].fillna(data['person_income'].median())
        
        if len(self.numerical_cols) > 0:
            data[self.numerical_cols] = self.min_max_scaler.transform(data[self.numerical_cols])
        
        if len(self.non_hierarchal_cat) > 0:
            cat_encoded = self.oh_encoder.transform(data[self.non_hierarchal_cat])
            cat_encoded_df = pd.DataFrame(
                cat_encoded,
                columns=self.oh_encoder.get_feature_names_out(self.non_hierarchal_cat),
                index=data.index
            )
            data = pd.concat([data.drop(columns=self.non_hierarchal_cat), cat_encoded_df], axis=1)
        
        if self.hierarchal_col in data.columns:
            data[self.hierarchal_col] = self.od_encoder.transform(data[[self.hierarchal_col]])
        
        return data
    
    def predict(self, data):
        processed_data = self.preprocess_data(data)
        predictions = self.model.predict(processed_data)
        
        result = []
        for pred in predictions:
            status = "Approved" if pred == 1 else "Rejected"
            result.append(status)
    
        return result

    def predict_proba(self, data):
        processed_data = self.preprocess_data(data)
        proba = self.model.predict_proba(processed_data)
        
        return proba
    
    def handle_outliers(self, data, method='cap'):
        data_clean = data.copy()
        
        for col in self.numerical_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            if method == 'cap':
                data_clean[col] = np.where(data_clean[col] < lower_bound, lower_bound, data_clean[col])
                data_clean[col] = np.where(data_clean[col] > upper_bound, upper_bound, data_clean[col])
        
        return data_clean
    
    def clean_gender(self, data):
        if 'person_gender' in data.columns:
            data['person_gender'] = data['person_gender'].str.lower()
            data['person_gender'] = data['person_gender'].replace({
                'fe male': 'female',
            })
        return data