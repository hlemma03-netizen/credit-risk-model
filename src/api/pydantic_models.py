from pydantic import BaseModel


class CustomerData(BaseModel):
    Total_Transaction_Amount: float
    Average_Transaction_Amount: float
    Transaction_Count: float
    Std_Transaction_Amount: float
    Avg_Transaction_Hour: float
    Avg_Transaction_Day: float
    Avg_Transaction_Month: float


class PredictionResponse(BaseModel):
    risk_probability: float
