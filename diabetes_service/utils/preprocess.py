import numpy as np
import joblib
import os
from huggingface_hub import hf_hub_download

# Get absolute path
#BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#scaler_path = os.path.join(BASE_DIR, 'model', 'scaler.pkl')
MODEL_REPO = "Ahmad10Raza/diabetes-service"
scaler_path = hf_hub_download(repo_id=MODEL_REPO, filename="scaler.pkl")
scaler = joblib.load(scaler_path)

# Categorical mappings
gender_map = {'Male': 0, 'Female': 1}
ethnicity_map = {'White': 0, 'Black': 1, 'Hispanic': 2, 'Other': 3}

def preprocess_input(data: dict):
    """
    Convert JSON input to model-ready numpy array.
    """
    try:
        gender = gender_map.get(data.get('gender', 'Male'), 0)
        ethnicity = ethnicity_map.get(data.get('ethnicity', 'Other'), 3)

        features = [
            gender,
            float(data.get('age', 0)),
            ethnicity,
            float(data.get('bmi', 0)),
            float(data.get('waist_circumference', 0)),
            float(data.get('systolic_bp', 0)),
            float(data.get('diastolic_bp', 0)),
            float(data.get('HbA1c', 0)),
            float(data.get('hdl_cholesterol', 0)),
            int(data.get('has_hypertension', 0)),
            int(data.get('takes_cholesterol_med', 0)),
            int(data.get('family_diabetes_history', 0))
        ]

        X = np.array(features).reshape(1, -1)
        X_scaled = scaler.transform(X)
        return X_scaled

    except Exception as e:
        raise ValueError(f"Invalid input data: {e}")
