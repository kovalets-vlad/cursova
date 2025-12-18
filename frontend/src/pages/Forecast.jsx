import React, { useState, useEffect, useMemo } from "react";
import { Brain, Heart, Activity, RefreshCw, AlertCircle } from "lucide-react";
import api from "../api/axiosConfig.js";
import { Card } from "../components/UI.jsx";
import { useStats } from "../context/StatsContext.jsx";
import { useAuth } from "../context/AuthContext.jsx";
import styles from "../module_styles/Forecast.module.css";

const Forecast = () => {
    const { stats, fetchStats, isLoaded } = useStats();
    const { user } = useAuth();

    const [predictLoading, setPredictLoading] = useState(false);
    const [prediction, setPrediction] = useState(null);
    const [aiAdvice, setAiAdvice] = useState(null);
    const [error, setError] = useState("");

    // Завантажуємо статистику в контекст, якщо її ще немає
    useEffect(() => {
        fetchStats();
    }, [fetchStats]);

    // Знаходимо останній запис у відсортованому масиві через useMemo
    const lastStat = useMemo(() => {
        if (!stats.length) return null;
        return [...stats].sort((a, b) => new Date(a.date) - new Date(b.date))[stats.length - 1];
    }, [stats]);

    const handlePredict = async () => {
        setPredictLoading(true);
        setPrediction(null);
        setAiAdvice(null);
        setError("");

        try {
            // 1. Отримуємо прогноз від ML моделі
            const res = await api.post("/api/predict_pulse");
            setPrediction(res.data);

            // 2. Отримуємо пораду від LLM (ШІ), використовуючи дані з контексту та профілю
            const adviceRes = await api.post("/api/prediction/advice", {
                user_stats: {
                    age: user?.age || 25, // Беремо реальний вік з профілю
                    stress_score: lastStat?.stress_score || 50,
                    minutesAsleep: lastStat?.minutesAsleep || 420,
                    sleep_efficiency: lastStat?.sleep_efficiency || 85,
                    steps: lastStat?.steps || 5000,
                    acwr: res.data.acwr_from_backend || 1.0,
                },
                prediction_delta: res.data.predicted_delta,
                predicted_bpm: res.data.predicted_bpm,
            });

            setAiAdvice(adviceRes.data.advice);
        } catch (e) {
            setError("Помилка прогнозу. Перевірте, чи достатньо даних (мінімум 5 днів).");
            console.error(e);
        } finally {
            setPredictLoading(false);
        }
    };

    return (
        <div className={styles.container}>
            {!prediction ? (
                <div className={styles.heroSection}>
                    <div className={styles.brainWrapper}>
                        <Brain className={styles.brainIcon} />
                    </div>
                    <h3 className={styles.heroTitle}>ШІ Аналіз Відновлення</h3>
                    <p className={styles.heroDescription}>
                        Нейромережа Pulse AI проаналізує твої показники за останні 7 днів, розрахує ACWR (навантаження)
                        та спрогнозує стан серцево-судинної системи.
                    </p>

                    <button
                        onClick={handlePredict}
                        disabled={predictLoading || stats.length < 5}
                        className={styles.predictBtn}
                    >
                        {predictLoading ? (
                            <span className={styles.loaderWrapper}>
                                <RefreshCw className={styles.spinner} size={20} />
                                Запускаю нейромережі...
                            </span>
                        ) : (
                            "Згенерувати прогноз"
                        )}
                    </button>

                    {stats.length < 5 && (
                        <div className={styles.warningBox}>
                            <AlertCircle size={16} />
                            <span>Потрібно ще {5 - stats.length} дн. даних для аналізу</span>
                        </div>
                    )}
                </div>
            ) : (
                <div className={styles.resultsWrapper}>
                    <div className={styles.statsRow}>
                        {/* КАРТКА ПРОГНОЗУ */}
                        <Card className={styles.predictionCard}>
                            <div className={styles.cardHeader}>
                                <Heart className="text-pink-500" size={20} />
                                <span>Прогноз пульсу спокою (RHR)</span>
                            </div>
                            <div className={styles.mainValueWrapper}>
                                <span className={styles.mainValue}>{prediction.predicted_bpm.toFixed(1)}</span>
                                <div
                                    className={`${styles.deltaBadge} ${
                                        prediction.predicted_delta > 0 ? styles.red : styles.green
                                    }`}
                                >
                                    {prediction.predicted_delta > 0 ? "↑" : "↓"}{" "}
                                    {Math.abs(prediction.predicted_delta).toFixed(1)}
                                </div>
                            </div>
                            <p className={styles.cardFooter}>Прогноз на завтра</p>
                        </Card>

                        {/* КАРТКА ВПЛИВУ ФАКТОРІВ */}
                        <Card className={styles.factorsCard}>
                            <h4 className={styles.smallTitle}>Вплив на результат (SHAP)</h4>
                            <div className={styles.factorsList}>
                                {Object.entries(prediction.details || {}).map(([key, val]) => (
                                    <div key={key} className={styles.factorItem}>
                                        <span className={styles.factorName}>{key.replace("_", " ")}</span>
                                        <span className={val > 0 ? styles.factorPlus : styles.factorMinus}>
                                            {val > 0 ? `+${val.toFixed(2)}` : val.toFixed(2)}
                                        </span>
                                    </div>
                                ))}
                            </div>
                        </Card>
                    </div>

                    {/* AI ADVICE */}
                    {aiAdvice && (
                        <Card className={styles.adviceCard}>
                            <div className={styles.adviceHeader}>
                                <div className={styles.adviceAvatar}>AI</div>
                                <h4>Рекомендація на основі аналізу</h4>
                            </div>
                            <div className={styles.adviceText}>{aiAdvice}</div>
                        </Card>
                    )}

                    <button onClick={() => setPrediction(null)} className={styles.resetBtn}>
                        <RefreshCw size={14} /> Новий аналіз
                    </button>
                </div>
            )}
        </div>
    );
};

export default Forecast;
