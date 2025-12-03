import joblib
from fastapi import FastAPI
from contextlib import asynccontextmanager
from keras.models import load_model
from .api import prediction

MODEL_PATH = 'cursova/models&scaller&features/'

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🔄 Завантаження моделей та скейлерів...")
    
    models_data = {}
    models_data['gru'] = load_model(f'{MODEL_PATH}gru3_delta_model.keras') 
    models_data['lstm'] = load_model(f'{MODEL_PATH}lstm3_delta_model.keras')
    models_data['cnn'] = load_model(f'{MODEL_PATH}cnn3_delta_model.keras')
    
    models_data['scaler_X'] = joblib.load(f'{MODEL_PATH}scaler_X.pkl')
    models_data['scaler_Y'] = joblib.load(f'{MODEL_PATH}scaler_Y.pkl')
    models_data['features'] = joblib.load(f'{MODEL_PATH}model_features.pkl')

    app.state.ml_models = models_data
    
    print("✅ Система готова до прогнозів!")
    yield

app = FastAPI(title="Cursova-Backend", lifespan=lifespan)

app.include_router(prediction.router, prefix="/api", tags=["Prediction"])