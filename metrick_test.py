import os

# 1. Налаштування змінних оточення для TensorFlow аби уникати попереджень
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['AUTOGRAPH_VERBOSITY'] = '0'

import tensorflow as tf
import logging

# 2. Ігнорування попереджень від TensorFlow
tf.get_logger().setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

import warnings

# 3. Ігнорування попереджень
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
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
models_info = {}

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.join(DIR_PATH, 'models_ensemble')
DATA_PATH = os.path.join(DIR_PATH, 'daily_fitbit_sema_df_processed.csv')
PLOTS_PATH = os.path.join(BASE_PATH, 'plots')
os.makedirs(PLOTS_PATH, exist_ok=True)

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
def load_models_and_scalers():
    print("🔄 Завантаження моделей та їх індивідуальних скейлерів...")
    for m_name in ['gru', 'lstm', 'cnn']:
        m_path = os.path.join(BASE_PATH, m_name)
        models_info[m_name] = {
            'model': load_model(os.path.join(m_path, f'final_{m_name}_model.keras')),
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
        ens_weighted = (model_predictions_bpm['gru'] * 0.25 + 
                        model_predictions_bpm['lstm'] * 0.25 + 
                        model_predictions_bpm['cnn'] * 0.5)
        
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

def save_separate_metrics_plots(report_df):
    # Налаштування стилю
    sns.set_theme(style="whitegrid")
    
    # --- ГРАФІК 1: R2 SCORE (Точність) ---
    plt.figure(figsize=(10, 6))
    ax1 = sns.barplot(x='Model', y='R2', hue='Set', data=report_df, palette='Blues_d')
    
    plt.title('Порівняння коефіцієнта детермінації ($R^2$)', fontsize=14, pad=15)
    plt.ylabel('Значення R^2 (0.0 - 1.0)')
    plt.xlabel('Модель')
    plt.ylim(0.8, 1.0) # Масштаб для наочності різниці
    
    # Додаємо цифри над стовпчиками
    for container in ax1.containers:
        ax1.bar_label(container, fmt='%.2f', padding=3)
    
    plt.tight_layout()
    fig_path = os.path.join(PLOTS_PATH, 'models_accuracy_r2.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close() # Закриваємо, щоб не виводити в консоль, а тільки зберегти
    print(f"✅ Графік R2 збережено як '{fig_path}'")

    # --- ГРАФІК 2: MAE (Помилка) ---
    plt.figure(figsize=(10, 6))
    ax2 = sns.barplot(x='Model', y='MAE', hue='Set', data=report_df, palette='Reds_d')
    
    plt.title('Порівняння середньої абсолютної помилки (MAE)', fontsize=14, pad=15)
    plt.ylabel('Помилка (BPM)')
    plt.xlabel('Модель')
    
    # Додаємо цифри над стовпчиками
    for container in ax2.containers:
        ax2.bar_label(container, fmt='%.2f', padding=3)
    
    plt.tight_layout()
    fig_path = os.path.join(PLOTS_PATH, 'models_error_mae.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Графік MAE збережено як '{fig_path}'")

def analyze_error_smoothing(user_ids, df, set_name="Test"):
    """
    Аналізує структуру помилок та ефект згладжування ансамблем.
    """
    print(f"🔍 Детальний аналіз помилок для вибірки: {set_name}...")
    all_data = []

    for u in user_ids:
        user_df = get_clean_user_df(df, u)
        if len(user_df) <= DAYS_WINDOW: continue

        y_real = user_df['resting_hr'].values[DAYS_WINDOW:]
        prev_bpm = user_df['resting_hr'].values[DAYS_WINDOW-1 : -1]
        
        # Отримуємо прогнози кожної моделі
        preds = {}
        for m_name, tools in models_info.items():
            dyn_scaled = tools['scaler_X'].transform(user_df[dynamic_cols].values)
            stat_data = user_df[static_cols].values.astype(float)
            stat_data[:, 0] /= 100.0; stat_data[:, 1] /= 50.0
            X_final = np.hstack((dyn_scaled, stat_data, user_df[['is_weekend']].values))
            X_wins = np.array([X_final[i : i + DAYS_WINDOW] for i in range(len(X_final) - DAYS_WINDOW)])
            
            p_z = tools['model'].predict(X_wins, verbose=0)
            p_delta = tools['scaler_Y'].inverse_transform(p_z).flatten()
            preds[m_name] = prev_bpm + p_delta

        # Ансамбль (середнє)
        ens_pred = (preds['gru'] + preds['lstm'] + preds['cnn']) / 3

        # Формуємо DataFrame для аналізу
        for i in range(len(y_real)):
            row_base = {"Actual_BPM": y_real[i], "User": u}
            for m in ['gru', 'lstm', 'cnn']:
                err = abs(y_real[i] - preds[m][i])
                all_data.append({**row_base, "Model": m.upper(), "Error": err, "Type": "Single"})
            
            ens_err = abs(y_real[i] - ens_pred[i])
            all_data.append({**row_base, "Model": "ENSEMBLE", "Error": ens_err, "Type": "Ensemble"})

    analysis_df = pd.DataFrame(all_data)

    # --- ГРАФІК 1: РОЗПОДІЛ ПОМИЛОК (VIOLIN PLOT) ---
    plt.figure(figsize=(12, 6))
    sns.violinplot(x='Model', y='Error', data=analysis_df, inner="quartile", palette="muted")
    plt.title(f'Розподіл помилок: Чи згладжує Ансамбль? ({set_name})')
    plt.ylabel('Абсолютна помилка (BPM)')
    plt.ylim(0, analysis_df['Error'].quantile(0.99)) # Відсікаємо викиди для наочності
    plt.savefig(os.path.join(PLOTS_PATH, 'error_distribution_violin.png'))
    plt.show()

    # --- ГРАФІК 2: ДЕ САМЕ МОДЕЛІ ПОМИЛЯЮТЬСЯ (Error vs Actual) ---
    plt.figure(figsize=(12, 6))
    # Використовуємо регресійну лінію, щоб побачити тренд помилки
    for m in ['LSTM', 'ENSEMBLE']:
        subset = analysis_df[analysis_df['Model'] == m]
        sns.regplot(x='Actual_BPM', y='Error', data=subset, scatter=False, label=f'Тренд помилки {m}')
    
    plt.title('Чи зростає помилка при зміні пульсу?')
    plt.xlabel('Реальний пульс (BPM)')
    plt.ylabel('Середня помилка')
    plt.legend()
    plt.savefig(os.path.join(PLOTS_PATH, 'error_trend_by_bpm.png'))
    plt.show()

    # Вивід статистики згладжування
    std_errors = analysis_df.groupby('Model')['Error'].std()
    print("\n📉 Стабільність моделей (Standard Deviation of Error):")
    print(std_errors.to_string())

def analyze_model_diversity(user_ids, df):
    print("🧠 Аналіз кореляції помилок (Diversity Analysis)...")
    residuals_data = {m: [] for m in ['gru', 'lstm', 'cnn']}
    
    for u in user_ids:
        user_df = get_clean_user_df(df, u) #
        if len(user_df) <= DAYS_WINDOW: continue

        y_real = user_df['resting_hr'].values[DAYS_WINDOW:]
        prev_bpm = user_df['resting_hr'].values[DAYS_WINDOW-1 : -1]

        for m_name, tools in models_info.items():
            # Стандартний процес отримання прогнозу (як у вашому коді)
            dyn_scaled = tools['scaler_X'].transform(user_df[dynamic_cols].values)
            stat_data = user_df[static_cols].values.astype(float)
            stat_data[:, 0] /= 100.0; stat_data[:, 1] /= 50.0
            X_final = np.hstack((dyn_scaled, stat_data, user_df[['is_weekend']].values))
            X_wins = np.array([X_final[i : i + DAYS_WINDOW] for i in range(len(X_final) - DAYS_WINDOW)])
            
            p_z = tools['model'].predict(X_wins, verbose=0)
            p_delta = tools['scaler_Y'].inverse_transform(p_z).flatten()
            pred_bpm = prev_bpm + p_delta
            
            # Розрахунок залишків (residuals)
            res = y_real - pred_bpm
            residuals_data[m_name].extend(res)

    # Створення DataFrame із залишками
    res_df = pd.DataFrame(residuals_data)
    corr_matrix = res_df.corr()

    # Візуалізація
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=0, vmax=1)
    plt.title('Кореляція помилок між моделями\n')
    plt.show()

    return corr_matrix

if __name__ == "__main__":
    load_models_and_scalers()
    df_full = pd.read_csv(DATA_PATH)
    all_users = df_full['id'].unique()
    train_users = all_users[:int(len(all_users)*0.8)]
    test_users = all_users[int(len(all_users)*0.8):]

    train_results = evaluate_on_users(train_users, "Train", df_full)
    test_results = evaluate_on_users(test_users, "Test", df_full)

    final_report = pd.concat([train_results, test_results]).groupby(['Model', 'Set']).mean().reset_index()
    final_report['MAE'] = final_report['MAE'].round(2)
    final_report['R2'] = final_report['R2'].round(2)
    final_report['RMSE'] = final_report['RMSE'].round(2)
    print("\n" + "="*60)
    print("ЗВІТ ТОЧНОСТІ МОДЕЛЕЙ")
    print("="*60)
    print(final_report.sort_values(by=['Model', 'Set']).to_string(index=False))

    save_separate_metrics_plots(final_report)
    analyze_error_smoothing(test_users, df_full, "Test")
    corr_res = analyze_model_diversity(test_users, df_full)
    print("\nКореляційна матриця помилок моделей:")
    print(corr_res.to_string())