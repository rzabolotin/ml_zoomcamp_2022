import bentoml
from bentoml.io import JSON
from pydantic import BaseModel

model_ref = bentoml.xgboost.get("credit_risk_model:lwpyyyssakoeh2ee")
dv = model_ref.custom_objects["dictVectorizer"]
my_runner = model_ref.to_runner()

svc = bentoml.Service("model_for_aws", runners=[my_runner])

class CreditRiskInput(BaseModel):
    seniority: int
    home: str
    time: int
    age: int
    marital: str
    records: str
    job: str
    expenses: int
    income: float
    assets: float
    debt: float
    amount: float
    price: float


@svc.api(input=JSON(pydantic_model=CreditRiskInput), output=JSON())
async def classify(input_dict: CreditRiskInput) -> dict:
    input_data = dv.transform(input_dict.dict())
    prediction = await my_runner.predict.async_run(input_data)
    approve = prediction[0] < 0.5
    return {
        "prediction": float(prediction[0]),
        "decision": "approve" if approve else "reject",
    }
