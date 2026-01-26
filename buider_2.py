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

def prepare_data_from_users(user_list, df, dynamic_cols, static_cols, scaler_X, scaler_Y, days_to_skip_at_end=0):
    X_list, y_list, base_hr_list = [], [], []
    for u in user_list:
        X_u, y_u_sc, raw_hr = process_user_with_delta(df, u, dynamic_cols, static_cols, scaler_X, scaler_Y)
        
        limit = len(X_u) - days_to_skip_at_end
        
        if limit > DAYS_WINDOW:
            for i in range(limit - DAYS_WINDOW):
                X_list.append(X_u[i : i + DAYS_WINDOW])
                y_list.append(y_u_sc[i + DAYS_WINDOW])
                base_hr_list.append(raw_hr[i + DAYS_WINDOW - 1])
                
    return np.array(X_list), np.array(y_list), np.array(base_hr_list)

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

    # --- КРОК 2: НАВЧАННЯ ТА АНАЛІЗ ---
    for m_type in MODEL_TYPES:
        # 1. Ініціалізація словника метрик 
        history_metrics = {'folds': [], 'train_mae': [], 'test_mae': [], 'test_r2': []}
        model_dir = os.path.join(BASE_OUTPUT_DIR, m_type.lower())
        os.makedirs(model_dir, exist_ok=True)
        
        test_group = user_folds[4]
        X_test, y_test, base_hr_test = prepare_data_from_users(test_group, df, dynamic_cols, static_cols, global_scaler_X, global_scaler_Y)

        for n_folds in range(1, 5):
            print(f"📊 Навчання на {n_folds} фолдах...")
            train_group = np.concatenate([user_folds[i] for i in range(n_folds)])
            X_train, y_train, base_hr_train = prepare_data_from_users(train_group, df, dynamic_cols, static_cols, global_scaler_X, global_scaler_Y)
            
            model = build_model((DAYS_WINDOW, X_train.shape[2]), m_type)
            model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
            
            # --- РОЗРАХУНОК TRAIN MAE (в BPM) ---
            train_preds_z = model.predict(X_train, verbose=0)
            train_pred_bpm = base_hr_train + global_scaler_Y.inverse_transform(train_preds_z).flatten()
            train_real_bpm = base_hr_train + global_scaler_Y.inverse_transform(y_train.reshape(-1, 1)).flatten()
            train_mae_val = mean_absolute_error(train_real_bpm, train_pred_bpm)
            
            # --- РОЗРАХУНОК TEST MAE (в BPM) ---
            test_preds_z = model.predict(X_test, verbose=0)
            test_pred_bpm = base_hr_test + global_scaler_Y.inverse_transform(test_preds_z).flatten()
            test_real_bpm = base_hr_test + global_scaler_Y.inverse_transform(y_test.reshape(-1, 1)).flatten()
            test_mae_val = mean_absolute_error(test_real_bpm, test_pred_bpm)
            test_r2_val = r2_score(test_real_bpm, test_pred_bpm)

            # ДОДАВАННЯ В СПИСКИ (Перевірте, щоб ці рядки виконувались на кожній ітерації)
            history_metrics['folds'].append(n_folds)
            history_metrics['train_mae'].append(train_mae_val)
            history_metrics['test_mae'].append(test_mae_val)
            history_metrics['test_r2'].append(test_r2_val)

            if n_folds == 4:
                X_final_test = X_test  # Збереження для SHAP аналізу пізніше
                final_model = model  # Збереження фінальної моделі
                final_model.save(os.path.join(model_dir, f'final_{m_type.lower()}_model.keras'))

        # --- ПОБУДОВА ГРАФІКА ---
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.set_xlabel('Кількість фолдів для навчання')
        ax1.set_ylabel('MAE (BPM)', color='black')
        
        # Основні лінії
        ax1.plot(history_metrics['folds'], history_metrics['test_mae'], 
                marker='o', color='tab:red', linewidth=2, label='Test MAE (BPM)')
        ax1.plot(history_metrics['folds'], history_metrics['train_mae'], 
                marker='s', color='tab:green', linestyle='--', linewidth=2, label='Train MAE (BPM)')
        
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        plt.title(f"Learning Curve: {m_type}")

        # 1. СПЕРШУ ЗБЕРІГАЄМО 
        save_path = os.path.join(model_dir, f'{m_type.lower()}_learning_curve.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Графік збережено: {save_path}")

        # 2. ПОТІМ ПОКАЗУЄМО
        plt.show()

        # 3. ЗАКРИВАЄМО 
        plt.close(fig)

        # # --- КРОК 3: SHAP АНАЛІЗ ---
        print(f"🔍 Розрахунок SHAP...")
        bg_idx = np.random.choice(X_final_test.shape[0], 100, replace=False)
        # Функція-обгортка для SHAP, яка розгортає 2D дані назад у 3D
        def predict_for_shap(x_flat):
            x_3d = x_flat.reshape(-1, DAYS_WINDOW, X_final_test.shape[2])
            return final_model.predict(x_3d, verbose=0)

        explainer = shap.KernelExplainer(predict_for_shap, shap.kmeans(X_final_test[bg_idx].reshape(100, -1), 10))
        test_idx = np.random.choice(X_final_test.shape[0], 30, replace=False)
        shap_values = explainer.shap_values(X_final_test[test_idx].reshape(30, -1))
        
        # Обробка виходу SHAP (для регресії це зазвичай список з одного масиву або просто масив)
        if isinstance(shap_values, list): sv = shap_values[0]
        else: sv = shap_values

        plt.figure(figsize=(10, 6))
        # Усереднюємо SHAP-значення за часовим вікном (3 дні)
        shap.summary_plot(np.mean(sv.reshape(-1, DAYS_WINDOW, X_final_test.shape[2]), axis=1), 
                          np.mean(X_final_test[test_idx], axis=1), 
                          feature_names=dynamic_cols+static_cols+weekend_col, show=False)
        plt.savefig(os.path.join(model_dir, 'shap_summary.png'))
        plt.close()

        # Збереження
        final_model.save(os.path.join(model_dir, f'{m_type.lower()}_model.keras'))
        joblib.dump(global_scaler_X, os.path.join(model_dir, 'scaler_X.pkl'))
        joblib.dump(global_scaler_Y, os.path.join(model_dir, 'scaler_Y.pkl'))
        joblib.dump(dynamic_cols+static_cols+weekend_col, os.path.join(model_dir, 'features.pkl'))
        print(f"✅ {m_type} готовo!")