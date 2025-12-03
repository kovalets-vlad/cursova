import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from keras.models import Sequential
from keras.layers import GRU, Dense, Dropout, Input, LSTM, Conv1D, Flatten, MaxPooling1D
from joblib import Parallel, delayed

dynamic_cols = [
    'steps', 'very_active_minutes', 'minutesAsleep', 'sleep_efficiency', 
    'nremhr', 'stress_score', 'nightly_temperature', 'resting_hr', 'is_weekend'
]
static_cols = ['age', 'bmi'] 
target_col = 'resting_hr'

# ==========================================
# 2. ДОПОМІЖНІ ФУНКЦІЇ
# ==========================================
def create_dataset(dataset, target_index, time_steps=3):
    X, Y = [], []
    for i in range(len(dataset) - time_steps - 1):
        X.append(dataset[i:(i + time_steps), :])
        Y.append(dataset[i + time_steps, target_index])
    return np.array(X), np.array(Y)

def build_model(input_shape, model_type='GRU'):
    model = Sequential()
    model.add(Input(shape=input_shape))
    
    if model_type == 'GRU':
        model.add(GRU(64, return_sequences=True))
        model.add(Dropout(0.3))
        model.add(GRU(64))
        
    elif model_type == 'LSTM':
        model.add(LSTM(64, return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(64))
        
    elif model_type == 'CNN':
        # CNN любить бачити патерни
        model.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
        model.add(MaxPooling1D(pool_size=1)) # Можна прибрати, якщо вікно мале
        model.add(Flatten())
        model.add(Dense(50, activation='relu'))

    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def process_user_data(df, user_id, dynamic_cols, static_cols, scaler=None):
    user_df = df[df['id'] == user_id].copy().ffill().bfill()
    
    # 1. Динаміка
    dyn_data = user_df[dynamic_cols].values
    if scaler is None:
        scaler = StandardScaler()
        dyn_scaled = scaler.fit_transform(dyn_data)
    else:
        dyn_scaled = scaler.transform(dyn_data)
        
    # 2. Статика
    try:
        stat_data = user_df[static_cols].values
        stat_data[:, 0] = stat_data[:, 0] / 100.0 # Age
        stat_data[:, 1] = stat_data[:, 1] / 50.0  # BMI
    except KeyError:
        stat_data = np.zeros((len(user_df), len(static_cols)))

    final_data = np.hstack((dyn_scaled, stat_data))
    return final_data, scaler

# ==========================================
# 3. ФУНКЦІЯ ДЛЯ ПАРАЛЕЛЬНОГО ЗАПУСКУ
# ==========================================
def evaluate_single_fold(test_user, all_users, df, dynamic_cols, static_cols, target_idx):
    """
    Ця функція буде запускатися на окремому ядрі процесора.
    Вона тренує модель і повертає лише метрики (MAE, R2).
    """
    # Формуємо TRAIN
    train_users = [u for u in all_users if u != test_user]
    X_train_list, y_train_list = [], []
    
    for train_user in train_users:
        processed_data, _ = process_user_data(df, train_user, dynamic_cols, static_cols, scaler=None)
        X_u, y_u = create_dataset(processed_data, target_index=target_idx, time_steps=3)
        X_train_list.append(X_u)
        y_train_list.append(y_u)
        
    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    
    # Формуємо TEST
    test_data_processed, test_scaler = process_user_data(df, test_user, dynamic_cols, static_cols, scaler=None)
    X_test, y_test = create_dataset(test_data_processed, target_index=target_idx, time_steps=3)
    
    # Навчання (локальне для процесу)
    model = build_model((X_train.shape[1], X_train.shape[2]))
    # verbose=0 обов'язково, щоб не змішувалися принти з різних потоків
    model.fit(X_train, y_train, epochs=25, batch_size=16, verbose=0) 
    
    # Оцінка
    preds_scaled = model.predict(X_test, verbose=0)
    
    # Інверсія
    dummy_matrix = np.zeros((len(preds_scaled), len(dynamic_cols)))
    dummy_matrix[:, target_idx] = preds_scaled.flatten() 
    preds_real = test_scaler.inverse_transform(dummy_matrix)[:, target_idx]
    
    dummy_matrix[:, target_idx] = y_test.flatten()
    y_real = test_scaler.inverse_transform(dummy_matrix)[:, target_idx]
    
    mae = mean_absolute_error(y_real, preds_real)
    mse = mean_squared_error(y_real, preds_real)
    r2 = r2_score(y_real, preds_real)
    
    
    # Повертаємо ID юзера, щоб знати, чий це результат
    return test_user, mae, mse, r2

# ==========================================
# 4. ГОЛОВНИЙ БЛОК ВИКОНАННЯ
# ==========================================

if __name__ == "__main__":
    df = pd.read_csv('cursova/daily_fitbit_sema_df_processed.csv') 
    
    # ВИПРАВЛЕННЯ ТИПІВ (важливе для вашої помилки з діленням)
    df['age'] = pd.to_numeric(df['age'], errors='coerce').fillna(30) # Заповнюємо середнім, якщо пусто
    df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce').fillna(25)
    df['date'] = pd.to_datetime(df['date'])
    df['is_weekend'] = (df['date'].dt.dayofweek >= 5).astype(int)

    target_idx = dynamic_cols.index(target_col) 
    
    # Беремо ВСІХ юзерів
    all_users = df['id'].unique().tolist()
    np.random.shuffle(all_users) # Перемішуємо, щоб було чесно
    
    # Розбиваємо на 5 частин (Folds)
    n_folds = 5
    # np.array_split розріже список юзерів на 5 шматків
    user_folds = np.array_split(all_users, n_folds)
    
    print(f"Запускаємо 5-Fold CV на {len(all_users)} користувачах...")
    print(f"Кожен фолд містить ~{len(user_folds[0])} юзерів.")

    # Функція для одного фолда (тепер вона обробляє ГРУПУ юзерів)
    def evaluate_fold_group(fold_idx, folds, df, dynamic_cols, static_cols, target_idx):
        # Test users - це поточна група
        test_users_group = folds[fold_idx]
        
        # Train users - це всі інші групи разом
        train_users_group = np.concatenate([folds[i] for i in range(n_folds) if i != fold_idx])
        
        # 1. Підготовка TRAIN (велика купа даних)
        X_train_list, y_train_list = [], []
        for u in train_users_group:
            # scaler=None -> вчимо новий скейлер
            processed, _ = process_user_data(df, u, dynamic_cols, static_cols, scaler=None)
            X_u, y_u = create_dataset(processed, target_index=target_idx, time_steps=3)
            X_train_list.append(X_u)
            y_train_list.append(y_u)
        
        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)
        
        # 2. Навчання моделі
        model = build_model((X_train.shape[1], X_train.shape[2]), model_type='GRU')
        model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
        
        # 3. Тестування (на кожному юзері з тестової групи окремо)
        mae_scores = []
        mse_scores = []
        r2_scores = []
        
        for test_user in test_users_group:
            # Важливо: для тесту ми створюємо scaler на основі САМЕ ЦЬОГО юзера
            test_data, test_scaler = process_user_data(df, test_user, dynamic_cols, static_cols, scaler=None)
            X_test, y_test = create_dataset(test_data, target_index=target_idx, time_steps=3)
            
            if len(X_test) == 0: continue # Пропускаємо, якщо у юзера мало даних
            
            preds_scaled = model.predict(X_test, verbose=0)
            
            # Інверсія
            dummy = np.zeros((len(preds_scaled), len(dynamic_cols)))
            dummy[:, target_idx] = preds_scaled.flatten()
            preds_real = test_scaler.inverse_transform(dummy)[:, target_idx]
            
            dummy[:, target_idx] = y_test.flatten()
            y_real = test_scaler.inverse_transform(dummy)[:, target_idx]
            
            mae_scores.append(mean_absolute_error(y_real, preds_real))
            mse_scores.append(mean_squared_error(y_real, preds_real))
            r2_scores.append(r2_score(y_real, preds_real))
            
        return np.mean(mae_scores), np.mean(mse_scores), np.mean(r2_scores)

    # Запуск паралельно (всього 5 задач, це буде швидко)
    # n_jobs=5 ідеально ляже на ваші 6 ядер
    results = Parallel(n_jobs=5)(
        delayed(evaluate_fold_group)(i, user_folds, df, dynamic_cols, static_cols, target_idx) 
        for i in range(n_folds)
    )
    
    # Вивід результатів
    final_mae = [res[0] for res in results]
    final_mse = [res[1] for res in results]
    final_r2 = [res[2] for res in results]
    
    print("\n" + "="*30)
    print(f"СЕРЕДНІЙ РЕЗУЛЬТАТ (5-Fold CV):")
    print(f"MAE: {np.mean(final_mae):.2f}")
    print(f"MSE: {np.mean(final_mse):.2f}")
    print(f"RMSE: {np.sqrt(np.mean(final_mse)):.2f}")
    print(f"R2: {np.mean(final_r2):.4f}")

# ==========================================
    # 5. ПІДГОТОВКА ДЛЯ SHAP ТА ЗБЕРЕЖЕННЯ
    # ==========================================
    print("\n🔄 Перенавчання фінальної моделі для SHAP та збереження...")
    
    # 1. Вибираємо дані (Strategy: Last Fold as Test)
    # user_folds[-1] - це масив ID користувачів з останньої групи
    test_users_group = user_folds[-1]
    
    # Всі інші групи об'єднуємо в один великий масив для навчання
    # np.concatenate склеїть 4 масиви в один довгий список
    train_users_group = np.concatenate(user_folds[:-1])
    
    print(f"Фінальне тренування: {len(train_users_group)} юзерів в Train, {len(test_users_group)} в Test")

    # 2. Підготовка TRAIN (проходимо по кожному юзеру з великої групи)
    X_train_list, y_train_list = [], []
    for train_user in train_users_group:
        # Обробляємо кожного юзера окремо
        processed_data, _ = process_user_data(df, train_user, dynamic_cols, static_cols, scaler=None)
        X_u, y_u = create_dataset(processed_data, target_index=target_idx, time_steps=3)
        X_train_list.append(X_u)
        y_train_list.append(y_u)
    
    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    
    # 3. Підготовка TEST (для SHAP і збереження скейлера)
    # Нам потрібен конкретний scaler для збереження, тому беремо ПЕРШОГО юзера з тестової групи
    target_test_user = test_users_group[0]
    
    test_data_processed, test_scaler = process_user_data(df, target_test_user, dynamic_cols, static_cols, scaler=None)
    X_test, y_test = create_dataset(test_data_processed, target_index=target_idx, time_steps=3)
    
    # 4. Фінальне навчання
    # Тут ми вже маємо правильні розмірності
    model = build_model((X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, epochs=25, batch_size=32, verbose=0)
    
    print("Рахуємо SHAP values...")
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    background_summary = shap.kmeans(X_train_flat, 20) 

    def predict_wrapper(data_flat):
        data_3d = data_flat.reshape(-1, predict_wrapper, len(dynamic_cols) + len(static_cols))
        return model.predict(data_3d, verbose=0)

    explainer = shap.KernelExplainer(predict_wrapper, background_summary)
    X_test_sample = X_test_flat[:50]
    shap_values = explainer.shap_values(X_test_sample)

    shap_values_3d = shap_values.reshape(-1, 3, len(dynamic_cols) + len(static_cols))
    shap_values_combined = np.sum(shap_values_3d, axis=1)
    
    X_test_sample_3d = X_test_sample.reshape(-1, 3, len(dynamic_cols) + len(static_cols))
    X_test_sample_combined = np.mean(X_test_sample_3d, axis=1)

    # Виправлений рядок з feature_names:
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_combined, X_test_sample_combined, feature_names=dynamic_cols + static_cols)
    
    # Збереження
    model.save('old_my_heart_model.keras')
    joblib.dump(test_scaler, 'old_user_scaler.pkl')
    # Зберігаємо повний список фіч
    joblib.dump(dynamic_cols + static_cols, 'old_model_features.pkl') 
    print("✅ Все збережено успішно!")