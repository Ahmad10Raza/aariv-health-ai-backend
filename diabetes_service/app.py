from flask import Flask, request, jsonify
import joblib
from utils.preprocess import preprocess_input
import os
from huggingface_hub import hf_hub_download


app = Flask(__name__)

MODEL_REPO = "Ahmad10Raza/diabetes-service"
model_path = hf_hub_download(repo_id=MODEL_REPO, filename="diabetes_rf_model.pkl")
#BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # current folder
#model_path = os.path.join(BASE_DIR, 'model', 'diabetes_rf_model.pkl')
model = joblib.load(model_path)
# Load trained Random Forest
#model = joblib.load('diabetes_service/models/diabetes_rf_model.pkl')

@app.route('/')
def home():
    return "Diabetes Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        X = preprocess_input(data)
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0][1]  # Probability of diabetes
        
        response = {
            'prediction': int(pred),  # 0 = Not Diabetes, 1 = Diabetes
            'probability': float(proba)
        }
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
