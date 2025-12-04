from fastapi import APIRouter, Request, HTTPException, Depends
import pandas as pd
import numpy as np
from utils.request_utils import generate_user_advice
from schemas.health_model import HealthDataRequest
from db.dependencies import get_current_user 
from db.session import AsyncSessionDep
from db.models_db import User, DailyStats
from sqlalchemy import select

router = APIRouter()

DAYS_WINDOW = 3 
LIMIT_DAYS_HISTORY = 50

@router.post("/predict_pulse")
async def predict_pulse(
    request: Request,
    session: AsyncSessionDep,
    current_user: User = Depends(get_current_user)
):

    query = select(DailyStats)\
        .where(DailyStats.user_id == current_user.id)\
        .order_by(DailyStats.date.desc())\
        .limit(LIMIT_DAYS_HISTORY)
        
    result = await session.execute(query)
    stats_history = result.scalars().all()
    
    stats_history = stats_history[::-1]

    if len(stats_history) < DAYS_WINDOW:
        raise HTTPException(
            status_code=400, 
            detail=f"Недостатньо даних. Потрібно мінімум {DAYS_WINDOW} днів історії."
        )

    data_list = []
    for stat in stats_history:
        row = stat.model_dump() if hasattr(stat, 'model_dump') else stat.__dict__.copy()
 
        if row.get('age') is None: row['age'] = current_user.age or 30
        if row.get('bmi') is None: 
             h = (current_user.height or 175) / 100
             w = current_user.weight or 70
             row['bmi'] = round(w / (h*h), 2)
             
        data_list.append(row)

    df = pd.DataFrame(data_list)

    models = request.app.state.ml_models
    scaler_X = models['scaler_X']
    scaler_Y = models['scaler_Y']
    feature_names = models['features']
  
    df['acute_steps'] = df['steps'].rolling(window=7, min_periods=1).mean()
    df['chronic_steps'] = df['steps'].rolling(window=28, min_periods=1).mean()
    df['acwr'] = df['acute_steps'] / (df['chronic_steps'] + 1)
    
    try:
        _ = df[feature_names]
    except KeyError as e:
        if 'is_weekend' not in df.columns and 'date' in df.columns:
             df['is_weekend'] = df['date'].apply(lambda d: 1 if d.weekday() >= 5 else 0)
        try:
            _ = df[feature_names]
        except KeyError as e2:
            return {"error": f"Не вистачає колонки: {e2}"}
    
    # Підготовка X
    base_dynamic = [c for c in feature_names if c not in ['age', 'bmi', 'is_weekend']]
    new_features = ['acute_steps', 'chronic_steps', 'acwr']
    # Гарантуємо порядок колонок як при навчанні
    dynamic_cols = base_dynamic + [c for c in new_features if c not in base_dynamic]
    
    dyn_data = df[dynamic_cols].values
    dyn_scaled = scaler_X.transform(dyn_data)
    
    stat_data = df[['age', 'bmi']].values
    stat_data[:, 0] = stat_data[:, 0] / 100.0
    stat_data[:, 1] = stat_data[:, 1] / 50.0
    
    week_data = df[['is_weekend']].values
    final_input = np.hstack((dyn_scaled, stat_data, week_data))
    
    try:
        # Беремо останні DAYS_WINDOW днів (3 дні)
        X_window = final_input[-DAYS_WINDOW:].reshape(1, DAYS_WINDOW, final_input.shape[1])
    except ValueError:
         raise HTTPException(status_code=400, detail="Помилка формування вікна даних.")

    # Прогноз
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
        "user_id": current_user.id,
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