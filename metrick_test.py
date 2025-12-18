import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# ==========================================
# 1. НАЛАШТУВАННЯ
# ==========================================
DAYS_WINDOW = 3
dynamic_cols = [
    'steps', 'very_active_minutes', 'minutesAsleep', 'sleep_efficiency', 
    'nremhr', 'stress_score', 'nightly_temperature', 'resting_hr',
    'chronic_steps', 'acute_steps', 'acwr' 
]
static_cols = ['age', 'bmi']
weekend_col = ['is_weekend']
target_col = 'hr_delta'

BASE_PATH = 'cursova/models_ensemble/'
DATA_PATH = 'cursova/daily_fitbit_sema_df_processed.csv'

# ==========================================
# 2. ПІДГОТОВКА "ЧИСТИХ" ДАНИХ (БЕЗ СКЕЙЛИНГУ)
# ==========================================
def get_clean_user_df(df, user_id):
    """Готує DataFrame з усіма фічами, але БЕЗ нормалізації."""
    user_df = df[df['id'] == user_id].copy()
    
    # Створення вихідних
    if 'date' in user_df.columns:
        user_df['date'] = pd.to_datetime(user_df['date'])
        user_df['is_weekend'] = (user_df['date'].dt.dayofweek >= 5).astype(int)
    else:
        user_df['is_weekend'] = 0

    # Конвертація статики
    user_df['age'] = pd.to_numeric(user_df['age'], errors='coerce').fillna(30)
    user_df['bmi'] = pd.to_numeric(user_df['bmi'], errors='coerce').fillna(25)
    
    # Розрахунок навантаження
    user_df['chronic_steps'] = user_df['steps'].rolling(window=28, min_periods=1).mean()
    user_df['acute_steps'] = user_df['steps'].rolling(window=7, min_periods=1).mean()
    user_df['acwr'] = user_df['acute_steps'] / (user_df['chronic_steps'] + 1)
    user_df['hr_delta'] = user_df['resting_hr'].diff().fillna(0)
    
    user_df = user_df.ffill().bfill()
    user_df = user_df.iloc[1:].reset_index(drop=True)
    return user_df

# ==========================================
# 3. ЗАВАНТАЖЕННЯ АНСАМБЛЮ
# ==========================================
print("🔄 Завантаження моделей та їх індивідуальних скейлерів...")
models_info = {}
for m_name in ['gru', 'lstm', 'cnn']:
    m_path = os.path.join(BASE_PATH, m_name)
    models_info[m_name] = {
        'model': load_model(os.path.join(m_path, f'{m_name}_model.keras')),
        'scaler_X': joblib.load(os.path.join(m_path, 'scaler_X.pkl')),
        'scaler_Y': joblib.load(os.path.join(m_path, 'scaler_Y.pkl'))
        # 'model': load_model(f'cursova/models&scaller&features&test/{m_name}3_delta_model.keras'),
        # 'scaler_X': joblib.load(f'cursova/models&scaller&features&test/scaler_X.pkl'),
        # 'scaler_Y': joblib.load(f'cursova/models&scaller&features&test/scaler_Y.pkl')
    }
print("✅ Усі моделі та скейлери завантажено.")

# ==========================================
# 4. ОЦІНКА
# ==========================================
def evaluate_on_users(user_ids, set_name, df):
    print(f"📊 Обробка вибірки: {set_name}...")
    all_results = []
    
    for u in user_ids:
        # 1. Отримуємо чисті дані користувача
        user_df = get_clean_user_df(df, u)
        if len(user_df) <= DAYS_WINDOW: continue

        model_predictions_bpm = {}
        y_real_bpm = user_df['resting_hr'].values[DAYS_WINDOW:]

        # 2. Прогноз кожною моделлю зі своїм скейлером
        for m_name, tools in models_info.items():
            # Скейлинг входу саме для цієї моделі
            dyn_scaled = tools['scaler_X'].transform(user_df[dynamic_cols].values)
            
            stat_data = user_df[static_cols].values.astype(float)
            stat_data[:, 0] /= 100.0
            stat_data[:, 1] /= 50.0
            week_data = user_df[['is_weekend']].values
            
            X_final_user = np.hstack((dyn_scaled, stat_data, week_data))

            # Створення вікон
            X_wins = []
            for i in range(len(X_final_user) - DAYS_WINDOW):
                X_wins.append(X_final_user[i : i + DAYS_WINDOW])
            X_wins = np.array(X_wins)

            # Прогноз та зворотний скейлинг
            p_z = tools['model'].predict(X_wins, verbose=0)
            p_delta = tools['scaler_Y'].inverse_transform(p_z).flatten()
            
            # Розрахунок BPM (Пульс_вчора + Дельта_сьогодні)
            prev_bpm = user_df['resting_hr'].values[DAYS_WINDOW-1 : -1]
            model_predictions_bpm[m_name] = prev_bpm + p_delta

        # 3. Ансамблі
        ens_weighted = (model_predictions_bpm['gru'] * 0.33 + 
                        model_predictions_bpm['lstm'] * 0.33 + 
                        model_predictions_bpm['cnn'] * 0.34)
        
        preds_to_eval = {
            'GRU': model_predictions_bpm['gru'],
            'LSTM': model_predictions_bpm['lstm'],
            'CNN': model_predictions_bpm['cnn'],
            'Ens_Weighted': ens_weighted,
        }

        for name, pred in preds_to_eval.items():
            all_results.append({
                "Model": name, 
                "Set": set_name, 
                "MAE": mean_absolute_error(y_real_bpm, pred),
                "RMSE": np.sqrt(mean_squared_error(y_real_bpm, pred)),
                "R2": r2_score(y_real_bpm, pred)
            })

    return pd.DataFrame(all_results)

def evaluate_on_users_global(user_ids, set_name, df):
    print(f"📊 Глобальний розрахунок: {set_name}...")
    
    # Словники для накопичення всіх прогнозів та реальних значень
    all_preds = {m: [] for m in ['gru', 'lstm', 'cnn', 'ens_2']}
    all_actuals = []

    for u in user_ids:
        user_df = get_clean_user_df(df, u)
        if len(user_df) <= DAYS_WINDOW: continue

        model_predictions_bpm = {}
        y_real_bpm = user_df['resting_hr'].values[DAYS_WINDOW:]

        for m_name, tools in models_info.items():
            # Скейлинг та підготовка X_wins (як у вашому коді)
            dyn_scaled = tools['scaler_X'].transform(user_df[dynamic_cols].values)
            stat_data = user_df[static_cols].values.astype(float)
            stat_data[:, 0] /= 100.0
            stat_data[:, 1] /= 50.0
            X_final_user = np.hstack((dyn_scaled, stat_data, user_df[['is_weekend']].values))

            X_wins = np.array([X_final_user[i : i + DAYS_WINDOW] for i in range(len(X_final_user) - DAYS_WINDOW)])

            # Прогноз
            p_z = tools['model'].predict(X_wins, verbose=0)
            p_delta = tools['scaler_Y'].inverse_transform(p_z).flatten()
            
            prev_bpm = user_df['resting_hr'].values[DAYS_WINDOW-1 : -1]
            model_predictions_bpm[m_name] = prev_bpm + p_delta
            
            # Накопичуємо
            all_preds[m_name].extend(model_predictions_bpm[m_name])

        # Ансамбль GRU+LSTM
        ens_2 = (model_predictions_bpm['gru'] + model_predictions_bpm['lstm']) / 2
        all_preds['ens_2'].extend(ens_2)
        all_actuals.extend(y_real_bpm)

    # Розрахунок метрик по всьому накопиченому масиву
    results = []
    y_true = np.array(all_actuals)
    for name in all_preds:
        y_p = np.array(all_preds[name])
        results.append({
            "Model": name.upper(),
            "Set": set_name,
            "MAE": mean_absolute_error(y_true, y_p),
            "R2": r2_score(y_true, y_p)
        })
    return pd.DataFrame(results)

if __name__ == "__main__":
    df_full = pd.read_csv(DATA_PATH)
    all_users = df_full['id'].unique()
    train_users = all_users[:int(len(all_users)*0.8)]
    test_users = all_users[int(len(all_users)*0.8):]

    train_results = evaluate_on_users(train_users, "Train", df_full)
    test_results = evaluate_on_users(test_users, "Test", df_full)

    final_report = pd.concat([train_results, test_results]).groupby(['Model', 'Set']).mean().reset_index()
    print("\n" + "="*60)
    print("ЗВІТ ТОЧНОСТІ МОДЕЛЕЙ")
    print("="*60)
    print(final_report.sort_values(by=['Model', 'Set']).to_string(index=False))