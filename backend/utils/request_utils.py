import os
import google.generativeai as genai
from dotenv import load_dotenv

env_path = os.path.dirname(__file__) + '/.env'

load_dotenv(env_path)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Будь ласка, встановіть змінну оточення GOOGLE_API_KEY з вашим ключем API Google Generative AI.")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

def create_llm_prompt(user_stats, prediction_delta, predicted_bpm):
    """
    user_stats: словник з даними за останній день (stress, sleep, acwr...)
    prediction_delta: зміна пульсу (+1.5)
    predicted_bpm: фінальний пульс (58.0)
    """
    
    # 1. Визначаємо тон розмови
    trend = "погіршується" if prediction_delta > 0.5 else "покращується" if prediction_delta < -0.5 else "стабільний"
    
    # 2. Формуємо текст запиту
    prompt = f"""
    Ти — професійний спортивний фізіолог та тренер.
    Твоє завдання: Надати короткі, дієві поради користувачу на основі його біометричних даних.
    
    ОСНОВНІ ДАНІ:
    - Вік: {user_stats['age']} років
    - Прогноз на завтра: Пульс у спокої {predicted_bpm:.1f} уд/хв.
    - Тенденція: Стан {trend} (зміна {prediction_delta:+.2f} від норми).
    
    ФАКТОРИ ВПЛИВУ (за останню добу):
    - Стрес (0-100): {user_stats['stress_score']} (Норма < 40)
    - Сон: {user_stats['minutesAsleep']/60:.1f} годин (Ефективність {user_stats['sleep_efficiency']}%)
    - Навантаження (ACWR): {user_stats['acwr']:.2f} (Норма 0.8-1.3)
    - Кроки: {user_stats['steps']}
    
    ІНСТРУКЦІЯ:
    1. Поясни простими словами, чому пульс змінився (зв'яжи це зі сном, стресом або навантаженням).
    2. Дай 3 конкретні рекомендації на завтра (наприклад, "лягти спати до 22:00", "зробити дихальну вправу", "легка прогулянка замість бігу").
    3. Не використовуй медичні терміни, будь емпатичним.
    4. Якщо ACWR > 1.5, обов'язково попередь про ризик травми!
    
    Відповідь українською мовою:
    """
    return prompt

def get_ai_advice(prompt):
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Не вдалося отримати пораду від ШІ: {e}"

def generate_user_advice(user_stats, prediction_delta, predicted_bpm):
    prompt = create_llm_prompt(user_stats, prediction_delta, predicted_bpm)
    advice = get_ai_advice(prompt)
    return advice


