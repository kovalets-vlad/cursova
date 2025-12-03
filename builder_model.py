import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from keras.models import Sequential
from keras.layers import GRU, Dense, Dropout, Input, Conv1D, Flatten, MaxPooling1D, LSTM
from joblib import Parallel, delayed
import keras.backend as K

# ==========================================
# 1. НАЛАШТУВАННЯ
# ==========================================
# Вхідні дані
dynamic_cols = [
    'steps', 'very_active_minutes', 'minutesAsleep', 'sleep_efficiency', 
    'nremhr', 'stress_score', 'nightly_temperature', 'resting_hr',
    # НОВІ "ДОВГІ" ФІЧІ:
    'chronic_steps', 'acute_steps', 'acwr' 
]
# Статичні ознаки (не змінюються з часом)
static_cols = ['age', 'bmi']
# Інформація про вихідні дні
weekend_col = ['is_weekend']

# Цільова колонка - Delta (зміна пульсу)
target_col = 'hr_delta' 

# Розмір часового вікна для аналізу (кількість днів в історії)
DAYS_WINDOW = 3  # Модель дивиться на 3 днів назад для предикції

# ==========================================
# 2. ФУНКЦІЇ
# ==========================================

def create_dataset(dataset, target_index, time_steps=DAYS_WINDOW):
    """
    Створює слайдуючі вікна часових рядів.
    
    Args:
        dataset: Масив даних (n_samples, n_features)
        target_index: Індекс цільової змінної
        time_steps: Розмір вікна (DAYS_WINDOW днів)
    
    Returns:
        X: Масив форми (n_samples, time_steps, n_features) - входи для моделі
        Y: Масив цільових значень для кожного вікна
    """
    X, Y = [], []
    for i in range(len(dataset) - time_steps):
        X.append(dataset[i:(i + time_steps), :])
        Y.append(dataset[i + time_steps, target_index])
    return np.array(X), np.array(Y)

def build_model(input_shape, model_type='GRU'):
    """
    Будує нейронну мережу для предикції.
    
    Args:
        input_shape: Кортеж (DAYS_WINDOW, n_features) - форма входу
        model_type: Тип архітектури ('GRU', 'LSTM' або 'CNN')
    
    Returns:
        model: Скомпільована Keras модель
    
    Примітка:
        - GRU/LSTM хороші для послідовностей
        - CNN краща для локальних патернів
    """
    model = Sequential()
    model.add(Input(shape=input_shape))
    
    if model_type == 'GRU':
        # GRU шари для обробки часових рядів
        model.add(GRU(64, return_sequences=True))  # Повертає всю послідовність
        model.add(Dropout(0.3))  # Регуляризація (вимикає 30% нейронів)
        model.add(GRU(64))  # Фінальний шар повертає тільки останній вихід
        
    elif model_type == 'LSTM':
        # LSTM - альтернатива GRU з більшою пам'яттю
        model.add(LSTM(64, return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(64))
        
    elif model_type == 'CNN':
        # Згорткові шари для пошуку локальних патернів
        model.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
        model.add(MaxPooling1D(pool_size=1))
        model.add(Flatten())  # Розгортання в 1D вектор
        model.add(Dense(50, activation='relu'))

    # Фінальні шари для передикції однієї цінності (зміна пульсу)
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))  # Вихід - одне число (зміна в BPM)
    model.compile(optimizer='adam', loss='mse')  # MSE для регресії
    return model

def process_user_with_delta(df, user_id, dynamic_cols, static_cols, scaler_X=None, scaler_Y=None):
    """
    Препроцесує дані одного користувача.
    
    Args:
        df: DataFrame з усіма даними
        user_id: ID користувача
        dynamic_cols: Список динамічних ознак
        static_cols: Список статичних ознак
        scaler_X: Попередньо навчений scaler для X (якщо None - навчається на цьому користувачеві)
        scaler_Y: Попередньо навчений scaler для Y (цільова змінна)
    
    Returns:
        X_final: Масив ознак (n_days, n_features)
        y_scaled: Масштабована цільова змінна (зміна пульсу)
        scaler_X: Використаний scaler для X
        scaler_Y: Використаний scaler для Y
        raw_bpm: Початкові значення пульсу (для розрахунку абсолютних значень)
    """
    # Вибираємо дані користувача та заповнюємо пропуски
    user_df = df[df['id'] == user_id].copy()
    
    # 1. ОБЧИСЛЮЄМО "ДОВГІ" МЕТРИКИ (до видалення NaN)
    # Chronic Load (Хронічне навантаження) - середнє за 28 днів (4 тижні)
    # min_periods=1 дозволяє рахувати навіть на початку, поки немає 28 днів
    user_df['chronic_steps'] = user_df['steps'].rolling(window=28, min_periods=1).mean()
    
    # Acute Load (Гостре навантаження) - середнє за 7 днів
    user_df['acute_steps'] = user_df['steps'].rolling(window=7, min_periods=1).mean()
    
    # ACWR (Acute:Chronic Workload Ratio)
    # Додаємо +1 у знаменник, щоб уникнути ділення на 0, якщо юзер не ходив місяць
    user_df['acwr'] = user_df['acute_steps'] / (user_df['chronic_steps'] + 1)
    
    # 2. СТВОРЮЄМО DELTA (ЗМІНУ)
    user_df['hr_delta'] = user_df['resting_hr'].diff().fillna(0)
    
    # Заповнюємо пропуски, які могли виникнути (ffill/bfill)
    user_df = user_df.ffill().bfill()
    
    # Видаляємо перший рядок (бо там delta некоректна)
    user_df = user_df.iloc[1:].reset_index(drop=True)

    # 3. Обробка X (Вхідні дані)
    # Тепер input_features включає і нові колонки (acwr, chronic...), бо ми додали їх у dynamic_cols
    input_features = user_df[dynamic_cols].values
    
    # Скейлинг Динаміки (всіх 11 колонок)
    if scaler_X is None:
        scaler_X = StandardScaler()
        dyn_scaled = scaler_X.fit_transform(input_features)
    else:
        dyn_scaled = scaler_X.transform(input_features)
        
    # Скейлинг Статики + Weekend
    try:
        stat_data = user_df[static_cols].values
        stat_data[:, 0] = stat_data[:, 0] / 100.0 # Age
        stat_data[:, 1] = stat_data[:, 1] / 50.0  # BMI
        week_data = user_df[weekend_col].values
    except KeyError:
        stat_data = np.zeros((len(user_df), 2))
        week_data = np.zeros((len(user_df), 1))
        
    X_final = np.hstack((dyn_scaled, stat_data, week_data))
    
    # 4. Обробка Y (Тільки Delta)
    target_values = user_df[[target_col]].values
    if scaler_Y is None:
        scaler_Y = StandardScaler()
        y_scaled = scaler_Y.fit_transform(target_values)
    else:
        y_scaled = scaler_Y.transform(target_values)
        
    raw_bpm = user_df['resting_hr'].values
        
    return X_final, y_scaled, scaler_X, scaler_Y, raw_bpm

# ==========================================
# 3. ОСНОВНИЙ ЦИКЛ (5-Fold Cross-Validation)
# ==========================================
# 5-Fold CV розділяє користувачів на 5 груп дляефективної оцінки
if __name__ == "__main__":
    # Завантажуємо та препроцесуємо дані
    df = pd.read_csv('cursova/daily_fitbit_sema_df_processed.csv') 
    df['age'] = pd.to_numeric(df['age'], errors='coerce').fillna(30)  # Заповнюємо пропуски
    df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce').fillna(25)
    df['date'] = pd.to_datetime(df['date'])
    # Визначаємо вихідні дні (субота=5, неділя=6)
    df['is_weekend'] = (df['date'].dt.dayofweek >= 5).astype(int)

    # Розділяємо користувачів для 5-Fold CV
    all_users = df['id'].unique().tolist()
    user_folds = np.array_split(all_users, 5)  # 5 групп користувачів
    
    print(f"🚀 Запуск Delta-Prediction (Window={DAYS_WINDOW} days)...")

    def evaluate_delta_fold(fold_idx, folds):
        """
        Оцінює модель на одному фолді CV.
        
        Args:
            fold_idx: Індекс тестового фолду (0-4)
            folds: Список всіх фолдів користувачів
        
        Returns:
            Кортеж (MAE, MSE, R2) - метрики на тестовому наборі
        """
        # Розділ 1: Підготовка тренувальної та тестової групи
        test_group = folds[fold_idx]  # 20% користувачів для тестування
        train_group = np.concatenate([folds[i] for i in range(5) if i != fold_idx])  # 80% для навчання
        
        # --- ПІДГОТОВКА TRAIN ДАНИХ ---
        X_train_list, y_train_list = [], []
        
        for u in train_group:
            # Обробляємо кожного користувача з тренувальної групи
            X_u, y_u_scaled, _, _, _ = process_user_with_delta(df, u, dynamic_cols, static_cols)
            
            # Створюємо слайдуючі вікна розміром DAYS_WINDOW
            # Для кожного вікна: X = 14 днів історії, Y = зміна пульсу на день 15
            X_wins, y_wins = [], []
            for i in range(len(X_u) - DAYS_WINDOW):
                X_wins.append(X_u[i : i + DAYS_WINDOW])  # 14 днів (4D вектор)
                y_wins.append(y_u_scaled[i + DAYS_WINDOW])  # Цінність на день 15
            
            if len(X_wins) > 0:
                X_train_list.append(np.array(X_wins))
                y_train_list.append(np.array(y_wins))
        
        # Об'єднуємо всі вікна від усіх користувачів
        X_train = np.concatenate(X_train_list, axis=0)  # Форма: (n_windows, 14, n_features)
        y_train = np.concatenate(y_train_list, axis=0)  # Форма: (n_windows,)
        
        # --- НАВЧАННЯ МОДЕЛІ ---
        model = build_model((X_train.shape[1], X_train.shape[2]), model_type='GRU')
        model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
        
        # --- ТЕСТУВАННЯ НА КОЖНОМУ КОРИСТУВАЧІ ---
        mae_list, mse_list, r2_list = [], [], []
        
        for test_user in test_group:
            # Обробляємо тестового користувача
            X_u, y_u_scaled, sc_X, sc_Y, raw_bpm = process_user_with_delta(df, test_user, dynamic_cols, static_cols)
            
            X_test, y_test_scaled = [], []
            actual_prev_bpm = []   # Пульс в останній день вікна
            actual_future_bpm = []  # Реальний пульс наступного дня
            
            # Створюємо тестові вікна
            for i in range(len(X_u) - DAYS_WINDOW):
                X_test.append(X_u[i : i + DAYS_WINDOW])
                y_test_scaled.append(y_u_scaled[i + DAYS_WINDOW])
                
                # raw_bpm[i + DAYS_WINDOW - 1] = пульс на день 14 (останній день в вікні)
                actual_prev_bpm.append(raw_bpm[i + DAYS_WINDOW - 1]) 
                # raw_bpm[i + DAYS_WINDOW] = реальний пульс на день 15 (цільовий день)
                actual_future_bpm.append(raw_bpm[i + DAYS_WINDOW])
            
            if len(X_test) == 0: 
                continue
            
            # Робимо передикції
            X_test = np.array(X_test)
            pred_delta_z = model.predict(X_test, verbose=0)  # Передикована зміна (масштабована)
            pred_delta_bpm = sc_Y.inverse_transform(pred_delta_z).flatten()  # Повертаємо до оригіналу
            
            # Абсолютна передикція пульсу = пульс сьогодні + передикована зміна
            pred_final_bpm = np.array(actual_prev_bpm) + pred_delta_bpm
            y_real_bpm = np.array(actual_future_bpm)
            
            # Обчислюємо метрики точності
            mae_list.append(mean_absolute_error(y_real_bpm, pred_final_bpm))
            mse_list.append(mean_squared_error(y_real_bpm, pred_final_bpm))
            r2_list.append(r2_score(y_real_bpm, pred_final_bpm))
            
        return np.mean(mae_list), np.mean(mse_list), np.mean(r2_list)

    # Запуск 5-Fold CV паралельно (на всіх ядрах процесора)
    results = Parallel(n_jobs=5)(delayed(evaluate_delta_fold)(i, user_folds) for i in range(5))
    
    # Збираємо результати з усіх фолдів
    mae_final = [res[0] for res in results]
    mse_final = [res[1] for res in results]
    r2_final = [res[2] for res in results]
    
    # Виводимо усереднені результати
    print("\n" + "="*40)
    print(f"РЕЗУЛЬТАТИ DELTA-PREDICTION (Window={DAYS_WINDOW})")
    print("="*40)
    print(f"MAE: {np.mean(mae_final):.2f} BPM")  # Середня абсолютна помилка
    print(f"MSE: {np.mean(mse_final):.2f} BPM")  # Середня квадратична помилка
    print(f"RMSE: {np.mean(np.sqrt(mse_final)):.2f} BPM")  # Корінь MSE
    print(f"R2:  {np.mean(r2_final):.4f}")  # Коефіцієнт детермінації (0-1, чим вище тим краще)

    # ==========================================
    # 5. SHAP (Interpretability) & SAVE
    # ==========================================
    # SHAP аналізує вклад кожної ознаки в передикцію
    print("\n🔄 Перенавчання фінальної моделі для SHAP...")

    test_users_group = user_folds[-1]  # Остання група для тестування
    train_users_group = np.concatenate(user_folds[:-1])  # Перші 4 групи для навчання

    # 1. TRAIN PREP - як раніше
    X_train_list, y_train_list = [], []
    for train_user in train_users_group:
        X_u, y_u_scaled, _, _, _ = process_user_with_delta(df, train_user, dynamic_cols, static_cols)
        
        X_wins, y_wins = [], []
        for i in range(len(X_u) - DAYS_WINDOW):
            X_wins.append(X_u[i : i + DAYS_WINDOW])
            y_wins.append(y_u_scaled[i + DAYS_WINDOW])
            
        if len(X_wins) > 0:
            X_train_list.append(np.array(X_wins))
            y_train_list.append(np.array(y_wins))

    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)

    # 2. TEST PREP - один користувач для SHAP аналізу
    target_test_user = test_users_group[0]
    X_test_u, y_test_u, test_scaler_X, test_scaler_Y, _ = process_user_with_delta(df, target_test_user, dynamic_cols, static_cols)
    
    X_test_wins = []
    for i in range(len(X_test_u) - DAYS_WINDOW):
        X_test_wins.append(X_test_u[i : i + DAYS_WINDOW])
    X_test = np.array(X_test_wins)

    # 3. FINAL TRAIN - навчаємо нову модель на всіх тренувальних даних
    model = build_model((X_train.shape[1], X_train.shape[2]), model_type='GRU')
    model.fit(X_train, y_train, epochs=25, batch_size=32, verbose=0)

    # --- SHAP ANALYSIS ---
    print("Рахуємо SHAP values...")
    # Розгортаємо 3D дані в 2D для SHAP
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    
    # Список назв усіх ознак
    all_features = dynamic_cols + static_cols
    if 'is_weekend' in df.columns:
        all_features += ['is_weekend']

    # SHAP використовує підмножину даних як фон для порівняння (k-means спрощує)
    background_summary = shap.kmeans(X_train_flat, 20) 

    def predict_wrapper(data_flat):
        """
        Обгортка для передачі розгорнутих даних у модель.
        SHAP ока передає 2D дані, а модель очікує 3D (часові ряди).
        """
        n_features = X_train.shape[2]
        # Розгортаємо назад в 3D форму
        data_3d = data_flat.reshape(-1, DAYS_WINDOW, n_features)
        return model.predict(data_3d, verbose=0)

    # KernelExplainer більш універсальний але повільніший за DeepExplainer
    explainer = shap.KernelExplainer(predict_wrapper, background_summary)
    
    # Беремо перші 50 тестових вікон для аналізу
    X_test_sample = X_test[:50]
    X_test_sample_flat = X_test_sample.reshape(50, -1)
    
    # Розраховуємо SHAP values (вклад кожної ознаки)
    shap_values = explainer.shap_values(X_test_sample_flat)

    # Обробка SHAP output (може бути список для багатовиходу)
    n_features = X_train.shape[2]
    if isinstance(shap_values, list):
        shap_vals = shap_values[0]  # Беремо перший вихід
    else:
        shap_vals = shap_values

    # Розгортаємо SHAP values назад у 3D (часові ряди)
    shap_values_3d = shap_vals.reshape(-1, DAYS_WINDOW, n_features)
    # Усередняємо вклади по днях (сумуємо вклад кожної ознаки за усіма днями)
    shap_values_combined = np.sum(shap_values_3d, axis=1) 

    # Усередняємо дані по днях для візуалізації
    X_test_sample_3d = X_test_sample.reshape(-1, DAYS_WINDOW, n_features)
    X_test_sample_combined = np.mean(X_test_sample_3d, axis=1)

    # Візуалізуємо важливість ознак
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_combined, X_test_sample_combined, feature_names=all_features)
    
    # Збереження моделі та скалерів для подальшого використання
    print("\n💾 Збереження файлів...")
    model.save(f'gru{DAYS_WINDOW}_delta_model.keras')  # Збереженя як gru14_delta_model.keras
    joblib.dump(test_scaler_X, 'scaler_X.pkl')  # Для масштабування входу
    joblib.dump(test_scaler_Y, 'scaler_Y.pkl')  # Для зворотного масштабування виходу
    joblib.dump(all_features, 'model_features.pkl')  # Назви ознак для тестування
    print("✅ Все збережено успішно!")