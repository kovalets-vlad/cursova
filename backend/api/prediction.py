from fastapi import APIRouter, Request, HTTPException
import pandas as pd
import numpy as np
from ..utils.request_utils import generate_user_advice
from ..schemas.health_model import HealthDataRequest, PredictionRequest

router = APIRouter()

DAYS_WINDOW = 3 

@router.post("/predict_pulse")
def predict_pulse(input_data: PredictionRequest, request: Request):
    # 1. Валідація довжини історії
    # Нам треба мінімум 3 дні, щоб сформувати вікно для моделі
    if len(input_data.history) < DAYS_WINDOW:
        raise HTTPException(
            status_code=400, 
            detail=f"Потрібно мінімум {DAYS_WINDOW} дні історії для прогнозу!"
        )

    # 2. Отримуємо ресурси
    models = request.app.state.ml_models
    scaler_X = models['scaler_X']
    scaler_Y = models['scaler_Y']
    feature_names = models['features']
    
    # 3. Конвертація в DataFrame
    data = [day.dict() for day in input_data.history]
    df = pd.DataFrame(data)
    
    # 4. Розрахунок Feature Engineering (ACWR)
    df['acute_steps'] = df['steps'].rolling(window=7, min_periods=1).mean()
    df['chronic_steps'] = df['steps'].rolling(window=28, min_periods=1).mean()
    df['acwr'] = df['acute_steps'] / (df['chronic_steps'] + 1)
    
    # 5. Підготовка X (Вхідні дані)
    try:
        # Перевірка наявності всіх колонок
        _ = df[feature_names]
    except KeyError as e:
        return {"error": f"Не вистачає колонки в даних: {e}"}
    
    # Розділяємо на групи для правильного скейлингу
    dynamic_cols = [c for c in feature_names if c not in ['age', 'bmi', 'is_weekend']]
    
    # Скейлинг динаміки
    dyn_data = df[dynamic_cols].values
    dyn_scaled = scaler_X.transform(dyn_data)
    
    # Скейлинг статики (ручний)
    stat_data = df[['age', 'bmi']].values
    stat_data[:, 0] = stat_data[:, 0] / 100.0
    stat_data[:, 1] = stat_data[:, 1] / 50.0
    
    # Вихідні
    week_data = df[['is_weekend']].values
    
    # Об'єднання в одну матрицю
    final_input = np.hstack((dyn_scaled, stat_data, week_data))
    
    # 6. Формування вікна (3 дні)
    try:
        # Беремо останні N рядків (DAYS_WINDOW)
        X_window = final_input[-DAYS_WINDOW:].reshape(1, DAYS_WINDOW, final_input.shape[1])
    except ValueError:
         raise HTTPException(
             status_code=400, 
             detail=f"Помилка розмірності. Недостатньо даних для вікна {DAYS_WINDOW} днів."
         )
    
    # 7. Прогноз (Ансамбль)
    pred_gru = models['gru'].predict(X_window, verbose=0)[0][0]
    pred_lstm = models['lstm'].predict(X_window, verbose=0)[0][0]
    pred_cnn = models['cnn'].predict(X_window, verbose=0)[0][0]
    
    # Інверсія (Z-score -> Delta BPM)
    delta_gru = scaler_Y.inverse_transform([[pred_gru]])[0][0]
    delta_lstm = scaler_Y.inverse_transform([[pred_lstm]])[0][0]
    delta_cnn = scaler_Y.inverse_transform([[pred_cnn]])[0][0]
    
    # Середнє значення ансамблю
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
async def get_advice(request_data: HealthDataRequest):
    user_stats = request_data.user_stats
    prediction_delta = request_data.prediction_delta
    predicted_bpm = request_data.predicted_bpm
    
    advice = generate_user_advice(user_stats, prediction_delta, predicted_bpm)
    return {"advice": advice}