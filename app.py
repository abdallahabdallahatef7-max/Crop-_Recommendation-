from fastapi import FastAPI ,Request ,Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import joblib
import pandas as pd

app=FastAPI()

app.mount("/static",StaticFiles(directory="static"),name="static")
templates=Jinja2Templates(directory="templates")

model = joblib.load("xgboost_model.pkl")
encoder = joblib.load("label_encoder.pkl")

@app.get("/",response_class=HTMLResponse)
async def home(request:Request):
    return templates.TemplateResponse("index.html",{"request":request ,"result":None})

@app.post("/predict",response_class=HTMLResponse)
async def predict(
    request:Request,
    N:float=Form(...)
    ,P:float=Form(...),
    K:float=Form(...),
    temperature: float = Form(...),
    humidity: float = Form(...),
    ph: float = Form(...),
    rainfall: float = Form(...)
):
    data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]], 
                        columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
    
    prediction = model.predict(data)
    crop_name = encoder.inverse_transform(prediction)[0]
    
    return templates.TemplateResponse("index.html", {"request": request, "result": crop_name})
