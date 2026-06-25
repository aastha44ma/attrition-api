from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS so Next.js can talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the saved assets
model = joblib.load('gb_attrition_model.pkl')
scaler = joblib.load('scaler.pkl')
expected_features = joblib.load('feature_columns.pkl')

class EmployeeData(BaseModel):
    MonthlyIncome: float
    Age: int
    OverTime: str # "Yes" or "No"
    YearsAtCompany: int
    # Add other top features you want the HR director to input

@app.post("/predict")
def predict_attrition(data: EmployeeData):
    # 1. Convert incoming JSON to a DataFrame
    input_df = pd.DataFrame([data.dict()])
    
    # 2. Create a blank dataframe with the exact columns the model expects
    model_input = pd.DataFrame(columns=expected_features)
    model_input.loc[0] = 0 # Fill with 0s initially
    
    # 3. Map the user inputs into the correct columns
    model_input['MonthlyIncome'] = input_df['MonthlyIncome']
    model_input['Age'] = input_df['Age']
    model_input['YearsAtCompany'] = input_df['YearsAtCompany']
    if input_df['OverTime'][0] == "Yes":
        model_input['OverTime_Yes'] = 1
        
    # 4. Scale the numeric columns just like in Colab
    # (Ensure you are scaling the exact same numeric features used in training)
    numeric_cols = ['MonthlyIncome', 'Age', 'YearsAtCompany'] 
    model_input[numeric_cols] = scaler.transform(model_input[numeric_cols])
    
    # 5. Predict!
    probability = model.predict_proba(model_input)[0][1]
    
    return {
        "flight_risk_score": round(probability * 100, 1),
        "risk_level": "High" if probability > 0.5 else "Low"
    }
