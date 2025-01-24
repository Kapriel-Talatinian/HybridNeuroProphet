from prophet import Prophet
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class TimeSeriesAnalyzer:
    def __init__(self):
        self.prophet_model = Prophet(
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0,
            seasonality_mode='multiplicative',
            daily_seasonality=True,
            weekly_seasonality=True
        )
        
    def prepare_data(self, prices, timestamps):
        """Prépare les données pour Prophet"""
        df = pd.DataFrame({
            'ds': pd.to_datetime(timestamps),
            'y': prices
        })
        return df
        
    def analyze_series(self, prices, timestamps, periods_ahead=30):
        """Analyse complète de la série temporelle"""
        # Préparation des données
        df = self.prepare_data(prices, timestamps)
        
        # Entraînement du modèle Prophet
        self.prophet_model.fit(df)
        
        # Prédictions futures
        future = self.prophet_model.make_future_dataframe(periods=periods_ahead, freq='5min')
        forecast = self.prophet_model.predict(future)
        
        # Décomposition de la série
        components = self.prophet_model.predict(df)
        
        analysis = {
            'trend': components['trend'].values,
            'seasonal': components['seasonal'].values,
            'weekly': components['weekly'].values if 'weekly' in components else None,
            'daily': components['daily'].values if 'daily' in components else None,
            'forecast': forecast['yhat'].tail(periods_ahead).values,
            'forecast_upper': forecast['yhat_upper'].tail(periods_ahead).values,
            'forecast_lower': forecast['yhat_lower'].tail(periods_ahead).values,
            'changepoints': self.prophet_model.changepoints
        }
        
        return analysis
    
    def generate_signal(self, analysis, current_price):
        """Génère un signal de trading basé sur l'analyse"""
        forecast = analysis['forecast']
        forecast_upper = analysis['forecast_upper']
        forecast_lower = analysis['forecast_lower']
        
        # Calcul de la direction de la tendance
        trend_direction = np.polyfit(range(len(forecast)), forecast, 1)[0]
        
        # Calcul de la volatilité prévue
        volatility = np.mean(forecast_upper - forecast_lower)
        
        # Règles de trading
        if trend_direction > 0 and current_price < forecast_lower[0]:
            return "BUY", forecast[0]
        elif trend_direction < 0 and current_price > forecast_upper[0]:
            return "SELL", forecast[0]
        else:
            return "HOLD", forecast[0]