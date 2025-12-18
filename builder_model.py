import os
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from keras.models import Sequential
from keras.layers import GRU, Dense, Dropout, Input, Conv1D, LSTM, BatchNormalization, GlobalAveragePooling1D, Activation
from joblib import Parallel, delayed
from keras.optimizers import Adam

# ==========================================
# 1. НАЛАШТУВАННЯ
# ==========================================
dynamic_cols = ['steps', 'very_active_minutes', 'minutesAsleep', 'sleep_efficiency', 
                'nremhr', 'stress_score', 'nightly_temperature', 'resting_hr',
                'chronic_steps', 'acute_steps', 'acwr']
static_cols = ['age', 'bmi']
weekend_col = ['is_weekend']
target_col = 'hr_delta' 
DAYS_WINDOW = 3 

MODEL_TYPES = ['GRU', 'LSTM', 'CNN']
BASE_OUTPUT_DIR = 'cursova/models_ensemble'

# ==========================================
# 2. ФУНКЦІЇ МОДЕЛЕЙ ТА ОБРОБКИ
# ==========================================

def build_model(input_shape, model_type='GRU'):
    model = Sequential()
    model.add(Input(shape=input_shape))
    if model_type == 'GRU':
        model.add(GRU(64, return_sequences=True))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(GRU(64))
    elif model_type == 'LSTM':
        model.add(LSTM(128, return_sequences=True)) 
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(LSTM(64))
        model.add(BatchNormalization())

        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
    elif model_type == 'CNN':
        model.add(Conv1D(filters=64, kernel_size=2, padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv1D(filters=128, kernel_size=2, padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(GlobalAveragePooling1D())

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1)) 
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def process_user_with_delta(df, user_id, dynamic_cols, static_cols, scaler_X, scaler_Y):
    user_df = df[df['id'] == user_id].copy()
    user_df['age'] = pd.to_numeric(user_df['age'], errors='coerce').fillna(30)
    user_df['bmi'] = pd.to_numeric(user_df['bmi'], errors='coerce').fillna(25)
    
    # Розрахунок метрик
    user_df['chronic_steps'] = user_df['steps'].rolling(window=28, min_periods=1).mean()
    user_df['acute_steps'] = user_df['steps'].rolling(window=7, min_periods=1).mean()
    user_df['acwr'] = user_df['acute_steps'] / (user_df['chronic_steps'] + 1)
    user_df['hr_delta'] = user_df['resting_hr'].diff().fillna(0)
    
    user_df = user_df.ffill().bfill()
    user_df = user_df.iloc[1:].reset_index(drop=True)

    # Стандартизація через ГЛОБАЛЬНІ скалери
    dyn_scaled = scaler_X.transform(user_df[dynamic_cols].values)
    stat_data = user_df[static_cols].values.astype(float)
    stat_data[:, 0] /= 100.0
    stat_data[:, 1] /= 50.0
    X_final = np.hstack((dyn_scaled, stat_data, user_df[['is_weekend']].values))
    
    y_scaled = scaler_Y.transform(user_df[[target_col]].values)
    # ПОВЕРТАЄМО РІВНО 3 ЗНАЧЕННЯ
    return X_final, y_scaled, user_df['resting_hr'].values

# ==========================================
# 3. ОСНОВНИЙ ЦИКЛ НАВЧАННЯ
# ==========================================
if __name__ == "__main__":
    df = pd.read_csv('cursova/daily_fitbit_sema_df_processed.csv')
    df['date'] = pd.to_datetime(df['date'])
    df['is_weekend'] = (df['date'].dt.dayofweek >= 5).astype(int)

    # Попередня підготовка колонок для навчання скалерів
    df['chronic_steps'] = df.groupby('id')['steps'].transform(lambda x: x.rolling(window=28, min_periods=1).mean())
    df['acute_steps'] = df.groupby('id')['steps'].transform(lambda x: x.rolling(window=7, min_periods=1).mean())
    df['acwr'] = df['acute_steps'] / (df['chronic_steps'] + 1)
    df[['chronic_steps', 'acute_steps', 'acwr']] = df[['chronic_steps', 'acute_steps', 'acwr']].ffill().bfill()

    all_users = df['id'].unique().tolist()
    user_folds = np.array_split(all_users, 5) 

    # --- КРОК 1: ГЛОБАЛЬНІ СКЕЛЕРИ ---
    print("📏 Навчання глобальних скалерів...")
    global_scaler_X = StandardScaler().fit(df[dynamic_cols].values)
    all_deltas = df.groupby('id')['resting_hr'].diff().fillna(0).values.reshape(-1, 1)
    global_scaler_Y = StandardScaler().fit(all_deltas)

    # --- КРОК 2: НАВЧАННЯ МОДЕЛЕЙ ---
    for m_type in MODEL_TYPES:
        print(f"\n" + "="*50 + f"\n🏗️  МОДЕЛЬ: {m_type}\n" + "="*50)
        model_dir = os.path.join(BASE_OUTPUT_DIR, m_type.lower())
        os.makedirs(model_dir, exist_ok=True)

        def evaluate_fold(fold_idx):
            test_group = user_folds[fold_idx]
            train_group = np.concatenate([user_folds[i] for i in range(5) if i != fold_idx])
            
            X_tr_list, y_tr_list = [], []
            for u in train_group:
                X_u, y_u_sc, _ = process_user_with_delta(df, u, dynamic_cols, static_cols, global_scaler_X, global_scaler_Y)
                if len(X_u) > DAYS_WINDOW:
                    for i in range(len(X_u) - DAYS_WINDOW):
                        X_tr_list.append(X_u[i : i + DAYS_WINDOW])
                        y_tr_list.append(y_u_sc[i + DAYS_WINDOW])
            
            if not X_tr_list: return None
            
            X_tr_np, y_tr_np = np.array(X_tr_list), np.array(y_tr_list)
            model = build_model((DAYS_WINDOW, X_tr_np.shape[2]), m_type)
            model.fit(X_tr_np, y_tr_np, epochs=20, batch_size=32, verbose=0)
            
            # Оцінка
            metrics = {'MAE':[], 'MAPE':[], 'MSE':[], 'RMSE':[], 'R2':[]}
            for u in test_group:
                X_u, y_u_sc, raw_hr = process_user_with_delta(df, u, dynamic_cols, static_cols, global_scaler_X, global_scaler_Y)
                if len(X_u) <= DAYS_WINDOW: continue
                
                X_ts_wins = np.array([X_u[i:i+DAYS_WINDOW] for i in range(len(X_u)-DAYS_WINDOW)])
                pred_z = model.predict(X_ts_wins, verbose=0)
                pred_bpm = raw_hr[DAYS_WINDOW-1:-1] + global_scaler_Y.inverse_transform(pred_z).flatten()
                real_bpm = raw_hr[DAYS_WINDOW:]
                
                metrics['MAE'].append(mean_absolute_error(real_bpm, pred_bpm))
                metrics['MAPE'].append(mean_absolute_percentage_error(real_bpm, pred_bpm))
                metrics['MSE'].append(mean_squared_error(real_bpm, pred_bpm))
                metrics['RMSE'].append(np.sqrt(mean_squared_error(real_bpm, pred_bpm)))
                metrics['R2'].append(r2_score(real_bpm, pred_bpm))
            
            return {k: np.mean(v) for k, v in metrics.items()}

        # Запуск CV
        print(f"🧪 Запуск Cross-Validation...")
        cv_res = Parallel(n_jobs=5)(delayed(evaluate_fold)(i) for i in range(5))
        cv_res = [r for r in cv_res if r is not None]

        print(f"📈 AVG Metrics: MAE={np.mean([r['MAE'] for r in cv_res]):.2f}, MAPE={np.mean([r['MAPE'] for r in cv_res]):.2f}, MSE={np.mean([r['MSE'] for r in cv_res]):.2f}, RMSE={np.mean([r['RMSE'] for r in cv_res]):.2f}, R2={np.mean([r['R2'] for r in cv_res]):.4f}")

        # Фінальне навчання
        X_all, y_all = [], []
        for u in all_users:
            X_u, y_u_sc, _ = process_user_with_delta(df, u, dynamic_cols, static_cols, global_scaler_X, global_scaler_Y)
            if len(X_u) > DAYS_WINDOW:
                for i in range(len(X_u) - DAYS_WINDOW):
                    X_all.append(X_u[i : i + DAYS_WINDOW])
                    y_all.append(y_u_sc[i + DAYS_WINDOW])

        X_final_np, y_final_np = np.array(X_all), np.array(y_all)
        final_model = build_model((DAYS_WINDOW, X_final_np.shape[2]), m_type)
        final_model.fit(X_final_np, y_final_np, epochs=25, batch_size=32, verbose=0)

        # --- КРОК 3: SHAP АНАЛІЗ ---
        print(f"🔍 Розрахунок SHAP...")
        bg_idx = np.random.choice(X_final_np.shape[0], 100, replace=False)
        # Функція-обгортка для SHAP, яка розгортає 2D дані назад у 3D
        def predict_for_shap(x_flat):
            x_3d = x_flat.reshape(-1, DAYS_WINDOW, X_final_np.shape[2])
            return final_model.predict(x_3d, verbose=0)

        explainer = shap.KernelExplainer(predict_for_shap, shap.kmeans(X_final_np[bg_idx].reshape(100, -1), 10))
        test_idx = np.random.choice(X_final_np.shape[0], 30, replace=False)
        shap_values = explainer.shap_values(X_final_np[test_idx].reshape(30, -1))
        
        # Обробка виходу SHAP (для регресії це зазвичай список з одного масиву або просто масив)
        if isinstance(shap_values, list): sv = shap_values[0]
        else: sv = shap_values

        plt.figure(figsize=(10, 6))
        # Усереднюємо SHAP-значення за часовим вікном (3 дні)
        shap.summary_plot(np.mean(sv.reshape(-1, DAYS_WINDOW, X_final_np.shape[2]), axis=1), 
                          np.mean(X_final_np[test_idx], axis=1), 
                          feature_names=dynamic_cols+static_cols+weekend_col, show=False)
        plt.savefig(os.path.join(model_dir, 'shap_summary.png'))
        plt.close()

        # Збереження
        final_model.save(os.path.join(model_dir, f'{m_type.lower()}_model.keras'))
        joblib.dump(global_scaler_X, os.path.join(model_dir, 'scaler_X.pkl'))
        joblib.dump(global_scaler_Y, os.path.join(model_dir, 'scaler_Y.pkl'))
        joblib.dump(dynamic_cols+static_cols+weekend_col, os.path.join(model_dir, 'features.pkl'))
        print(f"✅ {m_type} готовo!")