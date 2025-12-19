import os
import pandas as pd

# Шляхи до файлів
CURRENT_FOLDER = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(CURRENT_FOLDER, "data")
DATASET_PATH = os.path.join(DATA_FOLDER, "daily_fitbit_sema_df_unprocessed.csv")
PROCESSED_DATASET_PATH = os.path.join(DATA_FOLDER, "daily_fitbit_sema_df_processed.csv")

# Колонки
COLUMNS_DIARY = {
    'raw': [
        'id', 'date', 'nightly_temperature', 'nremhr', 'rmssd',
        'minutesAsleep', 'minutesAwake',
        'steps', 'distance', 'calories',
        'moderately_active_minutes', 'sedentary_minutes', 'very_active_minutes',
        'bpm', 'resting_hr', 'daily_temperature_variation',
        'age', 'bmi'
    ],
    'derived': [
        'sleep_duration', 'sleep_efficiency',
        'exertion_points_percentage', 'stress_score'
    ]
}

TARGET_COLUMN = 'resting_hr'
STATIC_COLUMNS = ['id', 'date', 'age', 'bmi']


def load_data(path: str = DATASET_PATH) -> pd.DataFrame:
    print(f"Loading data from {path}")
    data = pd.read_csv(path)
    data['date'] = pd.to_datetime(data['date'])
    print("Total rows:", len(data))
    print("Total columns:", len(data.columns))
    print(data['id'].value_counts())
    return data


def filter_data(data: pd.DataFrame) -> pd.DataFrame:
    all_columns = COLUMNS_DIARY['raw'] + COLUMNS_DIARY['derived']
    filtered_data = data[[col for col in all_columns if col in data.columns]]
    print("Filtered columns:", list(filtered_data.columns))
    return filtered_data


def log_nan_statistics(df: pd.DataFrame, stage: str):
    print(f"\nNaN summary {stage}:")
    for col in df.columns:
        print(f"{col}: {df[col].isna().sum()} NaN")


def handle_nan_raw(data: pd.DataFrame, nan_threshold: float) -> pd.DataFrame:
    print("\nHandling NaN in raw features")
    total_rows = len(data)

    for col in COLUMNS_DIARY['raw']:
        if col not in data.columns:
            continue

        nan_count = data[col].isna().sum()
        nan_ratio = nan_count / total_rows
        if nan_count == 0:
            continue

        if nan_ratio > nan_threshold:
            data = data.drop(columns=[col])
            print(f"Dropped {col} ({nan_ratio:.1%} NaN)")
            continue

        if pd.api.types.is_numeric_dtype(data[col]):
            if "date" in data.columns and data[col].notna().sum() > 3:
                data.loc[:, col] = data[col].interpolate(method='linear', limit_direction='both')
            else:
                median_value = data[col].median()
                data.loc[:, col] = data[col].fillna(median_value)
        elif pd.api.types.is_object_dtype(data[col]) or pd.api.types.is_categorical_dtype(data[col]):
            mode_value = data[col].mode().iloc[0] if not data[col].mode().empty else "Unknown"
            data.loc[:, col] = data[col].fillna(mode_value)

    return data


def compute_derived(df: pd.DataFrame) -> pd.DataFrame:
    print("\nComputing derived features (only NaN values)")

    if 'minutesAsleep' in df.columns and 'minutesAwake' in df.columns:
        mask = df['sleep_duration'].isna() if 'sleep_duration' in df.columns else pd.Series(True, index=df.index)
        df.loc[mask, 'sleep_duration'] = df.loc[mask, 'minutesAsleep'] + df.loc[mask, 'minutesAwake']

    if 'minutesAsleep' in df.columns and 'sleep_duration' in df.columns:
        mask = df['sleep_efficiency'].isna() if 'sleep_efficiency' in df.columns else pd.Series(True, index=df.index)
        df.loc[mask, 'sleep_efficiency'] = df.loc[mask, 'minutesAsleep'] / df.loc[mask, 'sleep_duration']

    if 'moderately_active_minutes' in df.columns and 'very_active_minutes' in df.columns:
        mask = df['exertion_points_percentage'].isna() if 'exertion_points_percentage' in df.columns else pd.Series(True, index=df.index)
        df.loc[mask, 'exertion_points_percentage'] = (
            df.loc[mask, 'moderately_active_minutes'] * 0.5 + df.loc[mask, 'very_active_minutes']
        )

    if 'resting_hr' in df.columns and 'rmssd' in df.columns and 'sleep_efficiency' in df.columns:
        mask = df['stress_score'].isna() if 'stress_score' in df.columns else pd.Series(True, index=df.index)
        df.loc[mask, 'stress_score'] = (
            df.loc[mask, 'resting_hr'] / (df.loc[mask, 'rmssd'] + 1e-3)
        ) * (1 - df.loc[mask, 'sleep_efficiency'])

    return df


def save_data(data: pd.DataFrame, file_path: str) -> None:
    data.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")


def split_data(data: pd.DataFrame, test_size: float = 0.2):
    train_list = []
    test_list = []

    for user_id, user_df in data.groupby('id'):
        user_df = user_df.sort_values('date')
        cutoff_date = user_df['date'].quantile(1 - test_size)
        train_user = user_df[user_df['date'] <= cutoff_date]
        test_user = user_df[user_df['date'] > cutoff_date]
        train_list.append(train_user)
        test_list.append(test_user)

    train_df = pd.concat(train_list).reset_index(drop=True)
    test_df = pd.concat(test_list).reset_index(drop=True)
    print(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}")
    return train_df, test_df


def main():
    df = load_data()
    df_filtered = filter_data(df)

    log_nan_statistics(df_filtered, "before NaN handling")
    df_cleaned = handle_nan_raw(df_filtered, nan_threshold=0.7)
    log_nan_statistics(df_cleaned, "after raw NaN handling")
    df_final = compute_derived(df_cleaned)
    log_nan_statistics(df_final, "after derived computation")

    save_data(df_final, PROCESSED_DATASET_PATH)

    df_final = load_data(PROCESSED_DATASET_PATH)
    train_df, test_df = split_data(df_final, test_size=0.2)
    save_data(train_df, os.path.join(DATA_FOLDER, "train_data.csv"))
    save_data(test_df, os.path.join(DATA_FOLDER, "test_data.csv"))


if __name__ == "__main__":
    main()
