from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np
import joblib
import logging

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ml-api")

# ---------- Load model on startup ----------
try:
    MODEL_BUNDLE = joblib.load("model.pkl")
    PIPELINE = MODEL_BUNDLE["pipeline"]
    FEATURE_NAMES = [fn.replace(" (cm)", "").replace(" ", "_") for fn in MODEL_BUNDLE["feature_names"]]
    TARGET_NAMES = MODEL_BUNDLE["target_names"]
    METRICS = MODEL_BUNDLE.get("metrics", {})
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.exception("Failed to load model.pkl")
    raise

# ---------- FastAPI app ----------
app = FastAPI(
    title="ML Model API - Iris",
    description="FastAPI for Iris species prediction",
    version="1.0.0"
)

# ---------- Schemas ----------
class PredictionInput(BaseModel):
    sepal_length: float = Field(..., ge=0, description="cm")
    sepal_width: float  = Field(..., ge=0, description="cm")
    petal_length: float = Field(..., ge=0, description="cm")
    petal_width: float  = Field(..., ge=0, description="cm")

class BatchPredictionInput(BaseModel):
    items: List[PredictionInput]

class PredictionOutput(BaseModel):
    prediction: str
    confidence: Optional[float] = None

class BatchPredictionOutput(BaseModel):
    predictions: List[PredictionOutput]

# ---------- Endpoints ----------
@app.get("/")
def health_check():
    return {"status": "healthy", "message": "ML Model API is running"}

@app.get("/model-info")
def model_info():
    return {
        "model_type": type(PIPELINE.named_steps["clf"]).__name__,
        "problem_type": "classification",
        "features": FEATURE_NAMES,
        "classes": TARGET_NAMES,
        "metrics": METRICS,
    }

def _predict_one(inp: PredictionInput) -> PredictionOutput:
    try:
        features = np.array([[inp.sepal_length, inp.sepal_width, inp.petal_length, inp.petal_width]])
        pred_idx = int(PIPELINE.predict(features)[0])
        pred_label = TARGET_NAMES[pred_idx]

        confidence = None
        if hasattr(PIPELINE, "predict_proba") or hasattr(PIPELINE.named_steps["clf"], "predict_proba"):
            probs = PIPELINE.predict_proba(features)[0]
            confidence = float(np.max(probs))

        return PredictionOutput(prediction=pred_label, confidence=confidence)
    except Exception as e:
        logger.exception("Prediction error")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    return _predict_one(input_data)

# Bonus: batch prediction
@app.post("/predict-batch", response_model=BatchPredictionOutput)
def predict_batch(batch: BatchPredictionInput):
    preds: List[PredictionOutput] = []
    for item in batch.items:
        preds.append(_predict_one(item))
    return BatchPredictionOutput(predictions=preds)
