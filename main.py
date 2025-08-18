from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
import os
import joblib
import pandas as pd
import numpy as np
from typing import Literal

# --- 1. Initialize FastAPI App ---
app = FastAPI(
    title="Credit Scoring Model API",
    description="API for predicting credit risk based on customer data. "
                "Higher scores (closer to 1) indicate higher credit risk.",
    version="1.0.0"
)

# --- 2. Global Variables for Model and Mappings ---
# These will be loaded once on startup
model_pipeline = None
categorical_mappings = None

MODEL_PATH = "credit_scoring_pipeline.pkl"
MAPPINGS_PATH = "categorical_mappings.pkl"

# --- 3. Load Model and Mappings on Startup ---
@app.on_event("startup")
async def load_resources():
    """
    Load the trained model pipeline and categorical mappings when the FastAPI application starts up.
    This ensures resources are loaded once and are available for all requests.
    """
    global model_pipeline, categorical_mappings
    
    # Load Model Pipeline
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
    try:
        model_pipeline = joblib.load(MODEL_PATH)
        print(f"Model pipeline loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model pipeline: {e}")
        raise RuntimeError(f"Could not load model pipeline: {e}")

    # Load Categorical Mappings
    if not os.path.exists(MAPPINGS_PATH):
        raise FileNotFoundError(f"Categorical mappings file not found at: {MAPPINGS_PATH}")
    try:
        categorical_mappings = joblib.load(MAPPINGS_PATH)
        print(f"Categorical mappings loaded successfully from {MAPPINGS_PATH}")
    except Exception as e:
        print(f"Error loading categorical mappings: {e}")
        raise RuntimeError(f"Could not load categorical mappings: {e}")


# --- 4. Define Input Schema (Pydantic Model) ---
# IMPORTANT: These fields and their types must match your training data columns.
# We use Literal for categorical fields to provide dropdowns in Swagger UI and ensure valid inputs.
class CreditApplicationInput(BaseModel):
    Age: int = Field(..., description="Age of the applicant in years (e.g., 25-70)")
    Income: float = Field(..., description="Annual income of the applicant (e.g., 50000.0)")
    MonthsEmployed: int = Field(..., description="Number of months the applicant has been employed (e.g., 12-500)")
    DTIRatio: float = Field(..., description="Debt-to-Income Ratio (e.g., 0.35)", ge=0.0, le=1.0) # Assuming 0 to 1
    Education: Literal['Primary', 'Secondary', 'Undergraduate', 'Postgraduate'] = Field(..., description="Highest level of education")
    EmploymentType: Literal['Unemployed', 'Salaried', 'Self-Employed', 'Contract-Part-time'] = Field(..., description="Type of employment")
    MaritalStatus: Literal['Single', 'Married', 'Divorced/Widowed'] = Field(..., description="Marital status")
    HasMortgage: int = Field(..., description="Does the applicant have a mortgage? (0=No, 1=Yes)", ge=0, le=1)
    HasDependents: int = Field(..., description="Does the applicant have dependents? (0=No, 1=Yes)", ge=0, le=1)
    LoanPurpose: Literal['Debt Consolidation', 'Home Improvement', 'Business', 'Education', 'Other-Miscellaneous'] = Field(..., description="Purpose of the loan")
    HasCoSigner: int = Field(..., description="Does the loan have a co-signer? (0=No, 1=Yes)", ge=0, le=1)

    # Example of a valid input payload (for documentation)
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "Age": 30,
                    "Income": 65000.0,
                    "MonthsEmployed": 120,
                    "DTIRatio": 0.40,
                    "Education": "Undergraduate",
                    "EmploymentType": "Salaried",
                    "MaritalStatus": "Married",
                    "HasMortgage": 1,
                    "HasDependents": 2,
                    "LoanPurpose": "Home Improvement",
                    "HasCoSigner": 0
                }
            ]
        }
    }

class PredictionOutput(BaseModel):
    """
    Schema for the output of the credit risk prediction.
    """
    credit_risk_prediction: Literal['Low Risk', 'High Risk'] = Field(..., description="Predicted credit risk: 'Low Risk' (0) or 'High Risk' (1)")
    probability_high_risk: float = Field(..., description="Probability of being 'High Risk' (closer to 1 indicates higher risk)", ge=0.0, le=1.0)


# --- 5. Define Prediction Endpoint ---
@app.post("/predict_credit_risk", response_model=PredictionOutput)
async def predict_credit_risk(data: CreditApplicationInput):
    """
    Predicts the credit risk (Low Risk or High Risk) for a given applicant.
    Returns the predicted risk category and the probability of being 'High Risk'.
    """
    if model_pipeline is None or categorical_mappings is None:
        raise HTTPException(status_code=503, detail="Model or mappings not loaded yet. Server is starting up.")

    try:
        # Convert Pydantic input to pandas DataFrame
        input_df = pd.DataFrame([data.model_dump()]) # Use .model_dump() for Pydantic v2

        # --- Apply Categorical Mappings (matching your ML script) ---
        input_df['Education'] = input_df['Education'].map({v: k for k, v in categorical_mappings['education_mapping'].items()})
        input_df['EmploymentType'] = input_df['EmploymentType'].map({v: k for k, v in categorical_mappings['employment_type_mapping'].items()})
        input_df['MaritalStatus'] = input_df['MaritalStatus'].map({v: k for k, v in categorical_mappings['marital_status_mapping'].items()})
        input_df['LoanPurpose'] = input_df['LoanPurpose'].map({v: k for k, v in categorical_mappings['loan_purpose_mapping'].items()})
        
        # Convert mapped integer values back to object type for consistency with original df
        # This is crucial because OneHotEncoder expects object types for categorical columns
        for col in ['Education', 'EmploymentType', 'MaritalStatus', 'LoanPurpose']:
            input_df[col] = input_df[col].astype(str)


        # Ensure column order matches training (important for some pipelines, though ColumnTransformer handles it generally)
        # It's safer to provide columns in the order your pipeline expects, if your ColumnTransformer
        # was built on a DataFrame with a specific column order.
        # However, ColumnTransformer with named transformers and `select_dtypes` is usually robust to order.
        # The key is that the names of the columns (Age, Income, etc.) must match.
        
        # Get the prediction (0 or 1)
        prediction_numeric = model_pipeline.predict(input_df)[0]
        
        # Get the probability of the positive class (High Risk = 1)
        probability_high_risk = model_pipeline.predict_proba(input_df)[:, 1][0]

        # Map numeric prediction back to human-readable label
        credit_risk_label = "High Risk" if prediction_numeric == 1 else "Low Risk"

        return PredictionOutput(
            credit_risk_prediction=credit_risk_label,
            probability_high_risk=probability_high_risk
        )

    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}. Ensure input data is valid and model is loaded correctly.")

# --- 6. Health Check Endpoint ---
@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify the API is running and model/mappings are loaded.
    """
    model_loaded_status = "ready" if model_pipeline is not None else "loading_model"
    mappings_loaded_status = "ready" if categorical_mappings is not None else "loading_mappings"
    
    return {
        "status": "ok",
        "model_status": model_loaded_status,
        "mappings_status": mappings_loaded_status,
        "message": "API is operational"
    }

# --- 7. Run the Application ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 
