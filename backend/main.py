import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from keras.models import load_model
from api import prediction, auth, user, data
from db.init_db import create_db_and_tables

MODEL_PATH = 'models&scaller&features/'

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🔄 Ініціалізація бази даних...")
    await create_db_and_tables() 
    
    print("🔄 Завантаження моделей та скейлерів...")
    try:
        models_data = {}
        models_data['gru'] = load_model(f'{MODEL_PATH}gru_model.keras') 
        models_data['lstm'] = load_model(f'{MODEL_PATH}lstm_model.keras')
        models_data['cnn'] = load_model(f'{MODEL_PATH}cnn_model.keras')
        
        models_data['scaler_X'] = joblib.load(f'{MODEL_PATH}scaler_X.pkl')
        models_data['scaler_Y'] = joblib.load(f'{MODEL_PATH}scaler_Y.pkl')
        models_data['features'] = joblib.load(f'{MODEL_PATH}features.pkl')

        app.state.ml_models = models_data
        print("✅ Система готова і моделі в пам'яті!")
        
    except Exception as e:
        print(f"❌ Критична помилка завантаження моделей: {e}")
    
    yield 
    
    print("🛑 Очищення ресурсів...")
    if hasattr(app.state, 'ml_models'):
        app.state.ml_models.clear()


app = FastAPI(title="Cursova-Backend", lifespan=lifespan)

origins = [
    "http://localhost:5173",  
    "http://127.0.0.1:5173",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,      
    allow_credentials=True,      
    allow_methods=["*"],         
    allow_headers=["*"],         
)

app.include_router(prediction.router, prefix="/api", tags=["Prediction"])
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(user.router, prefix="/api", tags=["User"])
app.include_router(data.router, prefix="/api/stats", tags=["Data Management"])