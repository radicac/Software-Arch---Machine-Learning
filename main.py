import pickle
import string

from fastapi import FastAPI, HTTPException, Query
from joblib import load
from pydantic import BaseModel
from typing import List

app = FastAPI()

SVM = load('svm.ml')
LRG = load('linear.ml')

common_features = []

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/{model_type}/diabetis")
async def root(model_type: str, the_input: str = Query(None)):
    if model_type not in ["svm", "lin"]:
        raise HTTPException(status_code=400, detail="Not supported model " + model_type)

    features = list(map(lambda x: float(x), the_input.split(",")))
    if len(features) != 10:
        raise HTTPException(status_code=400, detail="The feature is composed of 10 characteristics, but" + str(len(features)) + " were given")
    print(features)
    if model_type == "lin":
        prediction = int(LRG.predict([features])[0])
    else:
        prediction = int(SVM.predict([features])[0])
    response = {"class": prediction}
    return response


@app.get("/common_features")
async def get_common_features():
    return {"common_features": common_features}


class FeaturesModel(BaseModel):
    features: List[float]


@app.put("/common_features")
async def add_common_features(features_model: FeaturesModel):
    if len(features_model.features) != 10:
        raise HTTPException(status_code=400, detail="The feature is composed of 10 characteristics, but" + str(len(features_model.features)) + " were given")
    print(features_model.features)
    common_features.append(features_model.features)

#curl -X PUT --header "Content-Type: application/json" --data "{\"id\": \"2\"}" http://localhost:8000/common_features