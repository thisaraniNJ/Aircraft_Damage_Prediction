from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib
import pandas as pd
import shap
import lime.lime_tabular
import numpy as np

app = FastAPI()

model = joblib.load("model.pkl")
columns = joblib.load("X_train_columns.pkl")

class AircraftData(BaseModel):
    data: dict  # e.g. {"altitude": 1234, "speed": 567, ...}

@app.post("/predict")
def predict(payload: AircraftData):
    input_df = pd.DataFrame([payload.data])
    pred = model.predict(input_df)[0]
    prob = float(model.predict_proba(input_df)[0][1])

    # SHAP (bar values only)
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(input_df)[1].tolist()[0]

    # LIME values (top 5 only)
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array([list(payload.data.values())] * 100),
        feature_names=columns,
        class_names=['No Damage', 'Damage'],
        mode='classification'
    )
    lime_exp = lime_explainer.explain_instance(
        data_row=input_df.iloc[0].values,
        predict_fn=model.predict_proba,
        num_features=5
    )
    lime_features = lime_exp.as_list()

    return {
        "prediction": int(pred),
        "probability": prob,
        "shap_values": dict(zip(columns, shap_vals)),
        "lime_explanation": lime_features
    }
