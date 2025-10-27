from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI
from pydantic import BaseModel


MODEL_PATH = Path(os.getenv("MODEL_PATH", "pipeline_v2.bin"))


def load_pipeline(path: Path = MODEL_PATH):
    with path.open("rb") as f:
        pipeline = pickle.load(f)
    return pipeline


def predict_proba(pipeline, record: Dict[str, Any]) -> float:
    proba = pipeline.predict_proba([record])[0, 1]
    return float(proba)


# ---------- FastAPI bits ----------
class LeadRequest(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float


# Load once at startup for both CLI and API use
PIPELINE = load_pipeline()

app = FastAPI(title="Lead Conversion Service", version="1.0.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: LeadRequest):
    record = req.model_dump()
    p = predict_proba(PIPELINE, record)
    return {
        "conversion_probability": p,
        "will_convert": p >= 0.5,
    }


# For question 3
# def main():
#     record = {
#         "lead_source": "paid_ads",
#         "number_of_courses_viewed": 2,
#         "annual_income": 79276.0,
#     }
#     p = predict_proba(PIPELINE, record)
#     print(json.dumps({"conversion_probability": p}))


# if __name__ == "__main__":
#     main()

# For Question 4
# My curl request
# curl -sS -X POST "http://127.0.0.1:9696/predict" \
#   -H "Content-Type: application/json" \
#   -d '{"lead_source":"organic_search","number_of_courses_viewed":4,"annual_income":80304.0}'

# For Question 5
# (homework) @brukeg ➜ /workspaces/machine-learning-zoomcamp/05-deployment/homework (main) $ docker image ls agrigorev/zoomcamp-model:2025
# REPOSITORY                 TAG       IMAGE ID       CREATED      SIZE
# agrigorev/zoomcamp-model   2025      4a9ecc576ae9   5 days ago   121MB

# For Question 6
# curl -sS "http://127.0.0.1:9696/predict" \
#   -H "Content-Type: application/json" \
#   -d '{"lead_source":"organic_search","number_of_courses_viewed":4,"annual_income":80304.0}'
