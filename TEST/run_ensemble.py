import pandas as pd
import numpy as np
import joblib
from keras.models import load_model
from request import generate_user_advice
# ==========================================
# 2. ЗАВАНТАЖЕННЯ АНСАМБЛЮ
# ==========================================
print("🔄 Завантаження РАДИ ДИРЕКТОРІВ (Ensemble)...")

# Завантажуємо 3 моделі (переконайся, що файли існують!)
model_gru = load_model('cursova/models&scaller&features/gru3_delta_model.keras')
model_lstm = load_model('cursova/models&scaller&features/lstm3_delta_model.keras') 
model_cnn = load_model('cursova/models&scaller&features/cnn3_delta_model.keras')  

scaler_X = joblib.load('scaler_X.pkl')
scaler_Y = joblib.load('scaler_Y.pkl')
feature_names = joblib.load('model_features.pkl')

# ==========================================
# 3. ДАНІ НОВОГО ЮЗЕРА
# ==========================================
# Сценарій: "Раптовий удар"
data_shock = {
    'steps': [10000, 10000, 10000, 10000, 10000, 10000, 1200], 
    'very_active_minutes': [40, 40, 40, 40, 40, 40, 0],
    'minutesAsleep': [480, 480, 480, 480, 480, 480, 120], 
    'sleep_efficiency': [95, 95, 95, 95, 95, 95, 20],
    'nremhr': [48, 48, 48, 48, 48, 48, 70], 
    'stress_score': [10, 10, 10, 10, 10, 10, 90], 
    'nightly_temperature': [36.6, 36.6, 36.6, 36.6, 36.6, 36.6, 38.5], 
    'resting_hr': [50, 50, 50, 50, 50, 50, 51], 
    'age': [30] * 7,
    'bmi': [24] * 7,
    'is_weekend': [0, 0, 0, 0, 1, 1, 0]
}
df = pd.DataFrame(data_shock)

# ==========================================
# 4. ПІДГОТОВКА ДАНИХ (Feature Engineering)
# ==========================================
print("⚙️ Розрахунок метрик...")

# Розрахунок ACWR (як при навчанні)
df['chronic_steps'] = df['steps'].rolling(window=28, min_periods=1).mean()
df['acute_steps'] = df['steps'].rolling(window=7, min_periods=1).mean()
df['acwr'] = df['acute_steps'] / (df['chronic_steps'] + 1)

# Формуємо вхід для моделі
# Список колонок має співпадати з тим, що в feature_names (з model_features.pkl)
# Але ми знаємо, що порядок: dynamic + static + weekend
dynamic_cols = [
    'steps', 'very_active_minutes', 'minutesAsleep', 'sleep_efficiency', 
    'nremhr', 'stress_score', 'nightly_temperature', 'resting_hr',
    'chronic_steps', 'acute_steps', 'acwr'
]
static_cols = ['age', 'bmi']
weekend_col = ['is_weekend']

# Скейлинг
dyn_data = df[dynamic_cols].values
dyn_scaled = scaler_X.transform(dyn_data)

stat_data = df[static_cols].values
stat_data[:, 0] = stat_data[:, 0] / 100.0
stat_data[:, 1] = stat_data[:, 1] / 50.0

week_data = df[weekend_col].values

# Об'єднання
final_input = np.hstack((dyn_scaled, stat_data, week_data))

# Вікно (3 днів)
DAYS_WINDOW = 3
if len(final_input) < DAYS_WINDOW:
    raise ValueError(f"Недостатньо даних! Треба мінімум {DAYS_WINDOW} днів.")

X_window = final_input[-DAYS_WINDOW:].reshape(1, DAYS_WINDOW, final_input.shape[1])

# ==========================================
# 5. ПРОГНОЗ АНСАМБЛЕМ
# ==========================================
print("🧠 Голосування моделей...")

# Прогнози в Z-scores
z_gru = model_gru.predict(X_window, verbose=0)[0][0]
z_lstm = model_lstm.predict(X_window, verbose=0)[0][0]
z_cnn = model_cnn.predict(X_window, verbose=0)[0][0]

# Інверсія в BPM
d_gru = scaler_Y.inverse_transform([[z_gru]])[0][0]
d_lstm = scaler_Y.inverse_transform([[z_lstm]])[0][0]
d_cnn = scaler_Y.inverse_transform([[z_cnn]])[0][0]

# Середнє
delta_ensemble = (d_gru + d_lstm + d_cnn) / 3

today_bpm = df['resting_hr'].iloc[-1]
final_bpm = today_bpm + delta_ensemble

# Звіт
print("\n" + "="*50)
print(f"   АНСАМБЛЕВИЙ ПРОГНОЗ (Consensus AI)")
print("="*50)
print(f"Поточний пульс (Today): {today_bpm:.1f} BPM")
print("-" * 50)
print(f"{'GRU':<10} | {d_gru:+.2f} BPM")
print(f"{'LSTM':<10} | {d_lstm:+.2f} BPM")
print(f"{'CNN':<10} | {d_cnn:+.2f} BPM")
print("-" * 50)
print(f"{'AVERAGE':<10} | {delta_ensemble:+.2f} BPM   | {final_bpm:.1f}  <-- ФІНАЛ")
print("="*50)

# ==========================================
# 6. ГЕНЕРАЦІЯ ПОРАДИ (GEMINI)
# ==========================================
print("\n🤖 ГЕНЕРАЦІЯ ПОРАДИ ВІД ШІ...")

# Збираємо дані за останній день для промпта
last_row = df.iloc[-1]
user_stats = {
    'age': last_row['age'],
    'stress_score': last_row['stress_score'],
    'minutesAsleep': last_row['minutesAsleep'],
    'sleep_efficiency': last_row['sleep_efficiency'],
    'steps': last_row['steps'],
    'acwr': last_row['acwr']
}

# Створюємо запит
advice_text = generate_user_advice(user_stats, delta_ensemble, final_bpm)

print("-" * 60)
print(advice_text)
print("-" * 60)