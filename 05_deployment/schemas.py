from pydantic import BaseModel

class ClientData(BaseModel):
    AMT_INCOME_TOTAL: float
    AMT_CREDIT: float
    DAYS_BIRTH: float
    DAYS_EMPLOYED: float

    bureau_credit_count: float
    bureau_credit_active_mean: float
    bureau_days_credit_mean: float
    bureau_amt_credit_sum: float
