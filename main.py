from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

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
    OverTime: str
    YearsAtCompany: int

@app.post("/predict")
def predict_attrition(data: EmployeeData):
    try:
        # 1. Create a dictionary with ALL expected features, defaulting to 0
        input_dict = {col: 0 for col in expected_features}
        
        # 2. Inject the data the HR Director just typed into the Next.js form
        input_dict['MonthlyIncome'] = data.MonthlyIncome
        input_dict['Age'] = data.Age
        input_dict['YearsAtCompany'] = data.YearsAtCompany
        
        if data.OverTime == "Yes" and 'OverTime_Yes' in input_dict:
            input_dict['OverTime_Yes'] = 1
            
        # 3. Convert to a Pandas DataFrame
        model_input = pd.DataFrame([input_dict])
        
        # 4. The Magic Fix: Ask the scaler exactly which columns it trained on, and scale those
        numeric_cols = scaler.feature_names_in_
        model_input[numeric_cols] = scaler.transform(model_input[numeric_cols])
        
        # 5. Predict the flight risk!
        probability = model.predict_proba(model_input)[0][1]
        
        return {
            "flight_risk_score": round(probability * 100, 1),
            "risk_level": "High" if probability > 0.5 else "Low"
        }

    except Exception as e:
        # If anything breaks, it will print to the Render logs and send the real error to Next.js
        print(f"Server Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
