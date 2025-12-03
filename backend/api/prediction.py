from fastapi import APIRouter, Request, HTTPException, Depends
import pandas as pd
import numpy as np
from utils.request_utils import generate_user_advice
from schemas.health_model import HealthDataRequest, PredictionRequest 
from db.dependencies import get_current_user 

router = APIRouter()

DAYS_WINDOW = 3 

@router.post("/predict_pulse")
def predict_pulse(
    input_data: PredictionRequest, 
    request: Request, 
    current_user: str = Depends(get_current_user)
):
    if len(input_data.history) < DAYS_WINDOW:
        raise HTTPException(
            status_code=400, 
            detail=f"Потрібно мінімум {DAYS_WINDOW} дні історії для прогнозу!"
        )

    models = request.app.state.ml_models
    scaler_X = models['scaler_X']
    scaler_Y = models['scaler_Y']
    feature_names = models['features']
    
    data = [day.dict() for day in input_data.history]
    df = pd.DataFrame(data)
    
    df['acute_steps'] = df['steps'].rolling(window=7, min_periods=1).mean()
    df['chronic_steps'] = df['steps'].rolling(window=28, min_periods=1).mean()
    df['acwr'] = df['acute_steps'] / (df['chronic_steps'] + 1)
    
    try:
        _ = df[feature_names]
    except KeyError as e:
        return {"error": f"Не вистачає колонки в даних: {e}"}
    
    base_dynamic = [c for c in feature_names if c not in ['age', 'bmi', 'is_weekend']]
    
    new_features = ['acute_steps', 'chronic_steps', 'acwr']
    
    dynamic_cols = base_dynamic + [c for c in new_features if c not in base_dynamic]
    
    dyn_data = df[dynamic_cols].values
    dyn_scaled = scaler_X.transform(dyn_data)
    
    stat_data = df[['age', 'bmi']].values
    stat_data[:, 0] = stat_data[:, 0] / 100.0
    stat_data[:, 1] = stat_data[:, 1] / 50.0
    
    week_data = df[['is_weekend']].values
    final_input = np.hstack((dyn_scaled, stat_data, week_data))
    
    try:
        X_window = final_input[-DAYS_WINDOW:].reshape(1, DAYS_WINDOW, final_input.shape[1])
    except ValueError:
         raise HTTPException(
             status_code=400, 
             detail=f"Помилка розмірності. Недостатньо даних."
         )

    pred_gru = models['gru'].predict(X_window, verbose=0)[0][0]
    pred_lstm = models['lstm'].predict(X_window, verbose=0)[0][0]
    pred_cnn = models['cnn'].predict(X_window, verbose=0)[0][0]
    
    delta_gru = scaler_Y.inverse_transform([[pred_gru]])[0][0]
    delta_lstm = scaler_Y.inverse_transform([[pred_lstm]])[0][0]
    delta_cnn = scaler_Y.inverse_transform([[pred_cnn]])[0][0]
    
    avg_delta = (delta_gru + delta_lstm + delta_cnn) / 3
    
    last_bpm = df['resting_hr'].iloc[-1]
    predicted_bpm = last_bpm + avg_delta
    
    return {
        "current_bpm": float(last_bpm),
        "predicted_delta": float(avg_delta),
        "predicted_bpm": float(predicted_bpm),
        "details": {
            "gru_prediction": float(delta_gru),
            "lstm_prediction": float(delta_lstm),
            "cnn_prediction": float(delta_cnn)
        },
        "status": "success"
    }

@router.post("/prediction/advice")
async def get_advice(
    request_data: HealthDataRequest, 
    current_user: str = Depends(get_current_user)
):
    
    user_stats = request_data.user_stats
    prediction_delta = request_data.prediction_delta
    predicted_bpm = request_data.predicted_bpm
    
    advice = generate_user_advice(user_stats, prediction_delta, predicted_bpm)
    return {"advice": advice}