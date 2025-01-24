from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import pandas as pd
import numpy as np
from prophet import Prophet
import uvicorn
import yfinance as yf
import joblib
import os
import asyncio
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

app = FastAPI(title="Prophet Trading API")
executor = ThreadPoolExecutor(max_workers=4)

class MarketData(BaseModel):
    price: float
    volume: float
    spread: float
    time: datetime

    class Config:
        schema_extra = {
            "example": {
                "price": 1.1234,
                "volume": 1000,
                "spread": 0.0001,
                "time": "2025-01-24T12:00:00"
            }
        }

class PredictionResponse(BaseModel):
    direction: str  # "Buy" ou "Sell"
    confidence: float

    class Config:
        schema_extra = {
            "example": {
                "direction": "Buy",
                "confidence": 0.85
            }
        }

class ProphetModel:
    def __init__(self):
        self.model = None
        self.last_training = None
        self.symbol = "EURUSD=X"
        self.history = pd.DataFrame()
        self.init_model()
    
    def init_model(self):
        self.model = Prophet(
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10,
            seasonality_mode='multiplicative',
            daily_seasonality=True,
            weekly_seasonality=True
        )
    
    async def update_data(self):
        try:
            print("Downloading market data...")
            data = await asyncio.get_event_loop().run_in_executor(
                executor,
                lambda: yf.download(self.symbol, start=(datetime.now() - pd.Timedelta(days=365)), interval='1h')
            )
            if not data.empty:
                self.history = pd.DataFrame({
                    'ds': pd.to_datetime(data.index).tz_localize(None),
                    'y': data['Close'].values.flatten()
                })
                print(f"Data downloaded successfully - Shape: {self.history.shape}")
                return True
            print("No data downloaded")
            return False
        except Exception as e:
            print(f"Data update error: {e}")
            return False

    async def train(self):
        try:
            print("Starting model training...")
            if await self.update_data():
                await asyncio.get_event_loop().run_in_executor(
                    executor,
                    self._train_model
                )
                self.last_training = datetime.now()
                print("Model trained successfully")
                return True
            return False
        except Exception as e:
            print(f"Training error: {e}")
            return False

    def _train_model(self):
        if not self.history.empty:
            print("Fitting Prophet model...")
            self.model.fit(self.history)
            joblib.dump(self.model, 'prophet_model.pkl')
            print("Model saved to disk")
        else:
            raise ValueError("No training data available")

    async def predict(self, market_data: MarketData) -> PredictionResponse:
        try:
            if (self.last_training is None or 
                (datetime.now() - self.last_training).total_seconds() > 86400):
                print("Model requires retraining...")
                await self.train()

            future_time = pd.to_datetime(market_data.time).tz_localize(None)
            future = pd.DataFrame({
                'ds': [future_time + pd.Timedelta(hours=1)]
            })
            
            print("Making prediction...")
            forecast = await asyncio.get_event_loop().run_in_executor(
                executor,
                lambda: self.model.predict(future)
            )
            
            current_price = market_data.price
            predicted_price = forecast.iloc[0]['yhat']
            confidence = abs(forecast.iloc[0]['yhat_upper'] - forecast.iloc[0]['yhat_lower'])
            confidence = 1 - (confidence / (current_price * 0.01))
            confidence = max(min(confidence, 1), 0)
            
            direction = "Buy" if predicted_price > current_price else "Sell"
            
            prediction = PredictionResponse(
                direction=direction,
                confidence=confidence
            )
            print(f"Prediction generated: Direction={direction}, Confidence={confidence:.2f}")
            return prediction
            
        except Exception as e:
            print(f"Prediction error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

# Initialisation du modèle global
prophet_model = ProphetModel()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up Prophet Trading API...")
    if os.path.exists('prophet_model.pkl'):
        try:
            print("Loading existing model...")
            prophet_model.model = joblib.load('prophet_model.pkl')
            prophet_model.last_training = datetime.now()
            print("Existing model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Training new model...")
            await prophet_model.train()
    else:
        print("No existing model found. Training new model...")
        await prophet_model.train()
    yield
    print("Shutting down Prophet Trading API...")

app = FastAPI(lifespan=lifespan)

@app.post("/predict", response_model=PredictionResponse)
async def predict(market_data: MarketData):
    print(f"Received prediction request for time: {market_data.time}")
    return await prophet_model.predict(market_data)

@app.post("/retrain")
async def retrain():
    print("Received retraining request")
    success = await prophet_model.train()
    if success:
        return {"status": "success", "message": "Model retrained successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to retrain model")

if __name__ == "__main__":
    print("Starting Prophet Trading Server...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=5000,
        workers=1,
        loop="asyncio"
    )