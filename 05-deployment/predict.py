import pickle
from fastapi import FastAPI
import uvicorn

from typing import Dict, Any

app = FastAPI(title="lead-convert-prediction")

with open('pipeline_v2.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)

def predict_single(client):
    result = pipeline.predict_proba(client)[0, 1]
    return float(result)

@app.post("/predict")
def predict(client: Dict[str, Any]):
    prob = predict_single(client)

    return {
         "convert_probability": prob,
         "convert": bool(prob >= 0.5)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)

