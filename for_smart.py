import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
import statsmodels
import pickle
import json as json

## Creating a Fastapi object
app = FastAPI()

sm_32_model = pickle.load(open('sm_32_model.sav', 'rb'))
data_32=pd.read_csv('test_sm_32.csv')
data_162=pd.read_csv('test_162.csv')
data_163=pd.read_csv('test_163.csv')


@app.post("/predict")
def predict():
    prediction_32 = results.predict(start=data_32.index[0],end=data_32.index[-1], exog=data_32)
    predict_32=pd.DataFrame(prediction_32, columns=['predicted_mean_32'])
    predicted_means_32 = predict_32[['predicted_mean_32']]
    
    prediction_162 = results.predict(start=data_162.index[0],end=data_162.index[-1], exog=data_162)
    predict_162=pd.DataFrame(prediction_162, columns=['predicted_mean_162'])
    predicted_means_162 = predict_162[['predicted_mean_162']]
    
    prediction_163 = results.predict(start=data_163.index[0],end=data_163.index[-1], exog=data_163)
    predict_163=pd.DataFrame(prediction_163, columns=['predicted_mean_163'])
    predicted_means_163 = predict_163[['predicted_mean_163']]
    
    prediction_32=predicted_means_32.to_json(orient='records')
    prediction_162=predicted_means_162.to_json(orient='records')
    prediction_163=predicted_means_163.to_json(orient='records')
    

   
    return prediction_32, prediction_162, prediction_163