import React, { useState, useEffect } from "react";
import { Brain, Heart, Activity } from "lucide-react";
import api from "../api/axiosConfig.js";
import { Card } from "../components/UI.jsx";

const Forecast = () => {
    const [stats, setStats] = useState([]);
    const [loading, setLoading] = useState(false);
    const [prediction, setPrediction] = useState(null);
    const [aiAdvice, setAiAdvice] = useState(null);
    const [error, setError] = useState("");

    // Нам потрібна статистика, щоб передати останні дані в ШІ для поради
    useEffect(() => {
        const loadStats = async () => {
            try {
                const res = await api.get("/api/stats/");
                setStats(res.data.sort((a, b) => new Date(a.date) - new Date(b.date)));
            } catch (e) {
                console.error("Помилка завантаження статистики для прогнозу", e);
            }
        };
        loadStats();
    }, []);

    const handlePredict = async () => {
        setLoading(true);
        setPrediction(null);
        setAiAdvice(null);
        setError("");

        try {
            // 1. Отримуємо прогноз (цифри)
            const res = await api.post("/api/predict_pulse");
            setPrediction(res.data);

            // 2. Отримуємо пораду від ШІ
            // Беремо останній запис зі статистики або дефолтні значення, якщо пусто
            const lastStat = stats[stats.length - 1] || {};

            const adviceRes = await api.post("/api/prediction/advice", {
                user_stats: {
                    age: 30, // Можна пізніше винести в профіль
                    stress_score: lastStat.stress_score || 50,
                    minutesAsleep: lastStat.minutesAsleep || 400,
                    sleep_efficiency: lastStat.sleep_efficiency || 85,
                    steps: lastStat.steps || 5000,
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
            setLoading(false);
        }
    };

    return (
        <div className="max-w-4xl mx-auto space-y-8 animate-in fade-in duration-500">
            {/* Header Section */}
            {!prediction ? (
                <div className="text-center py-10">
                    <div className="w-24 h-24 bg-slate-800 rounded-full flex items-center justify-center mx-auto mb-6 border border-slate-700 shadow-xl shadow-blue-900/20">
                        <Brain className="w-12 h-12 text-blue-500" />
                    </div>
                    <h3 className="text-2xl text-white font-bold mb-2">ШІ Аналіз Відновлення</h3>
                    <p className="text-slate-400 mb-8 max-w-lg mx-auto">
                        Нейромережа проаналізує твої показники за останні 7 днів та спрогнозує стан на завтра.
                    </p>

                    <button
                        onClick={handlePredict}
                        disabled={loading || stats.length < 5}
                        className="bg-blue-600 hover:bg-blue-700 text-white px-8 py-3 rounded-xl font-bold text-lg shadow-lg shadow-blue-600/20 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        {loading ? (
                            <span className="flex items-center gap-2">
                                <span className="animate-spin h-4 w-4 border-2 border-white/50 border-t-white rounded-full"></span>
                                Аналізую...
                            </span>
                        ) : (
                            "Згенерувати прогноз"
                        )}
                    </button>

                    {error && (
                        <p className="mt-4 text-red-400 text-sm bg-red-900/20 py-2 px-4 rounded-lg inline-block">
                            {error}
                        </p>
                    )}

                    {stats.length < 5 && !error && (
                        <p className="mt-4 text-yellow-500 text-sm">
                            * Потрібно ще {5 - stats.length} днів даних для точного прогнозу.
                        </p>
                    )}
                </div>
            ) : (
                /* Results Section */
                <div className="space-y-6 text-left">
                    <div className="grid md:grid-cols-2 gap-6">
                        {/* Prediction Card */}
                        <Card className="bg-gradient-to-br from-indigo-900/40 to-slate-900 border-indigo-500/30">
                            <div className="flex items-center gap-2 mb-4 text-indigo-300">
                                <Heart className="w-5 h-5" /> Прогноз пульсу спокою
                            </div>
                            <div className="flex items-baseline gap-3">
                                <span className="text-5xl font-bold text-white">
                                    {prediction.predicted_bpm.toFixed(1)}
                                </span>
                                <span
                                    className={`text-xl font-medium ${
                                        prediction.predicted_delta > 0 ? "text-red-400" : "text-green-400"
                                    }`}
                                >
                                    {prediction.predicted_delta > 0 ? "↗" : "↘"}{" "}
                                    {Math.abs(prediction.predicted_delta).toFixed(2)}
                                </span>
                            </div>
                            <div className="mt-4 text-slate-400 text-sm flex items-center gap-2">
                                <Activity className="w-4 h-4" /> Поточний: {stats[stats.length - 1]?.resting_hr} bpm
                            </div>
                        </Card>

                        {/* Details Card */}
                        <Card>
                            <div className="text-sm text-slate-400 mb-4 font-semibold uppercase tracking-wider">
                                Вплив факторів (SHAP)
                            </div>
                            <div className="space-y-3">
                                {Object.entries(prediction.details || {}).map(([key, val]) => (
                                    <div
                                        key={key}
                                        className="flex justify-between items-center p-2.5 bg-slate-900/50 rounded-lg border border-slate-800"
                                    >
                                        <span className="uppercase text-xs font-bold text-slate-500">
                                            {key.split("_").join(" ")}
                                        </span>
                                        <span
                                            className={`font-mono font-bold ${
                                                val > 0 ? "text-red-400" : "text-green-400"
                                            }`}
                                        >
                                            {val > 0 ? "+" : ""}
                                            {val.toFixed(2)}
                                        </span>
                                    </div>
                                ))}
                            </div>
                        </Card>
                    </div>

                    {/* AI Advice Card */}
                    {aiAdvice && (
                        <Card className="border-blue-500/30 bg-blue-900/10">
                            <div className="flex items-center gap-3 mb-4">
                                <div className="p-2 bg-blue-500 rounded-lg text-white shadow-lg shadow-blue-500/20">
                                    <Brain className="w-5 h-5" />
                                </div>
                                <h3 className="text-xl font-bold text-white">Рекомендація Тренера</h3>
                            </div>
                            <div className="prose prose-invert max-w-none text-slate-200 whitespace-pre-line leading-relaxed text-sm md:text-base">
                                {aiAdvice}
                            </div>
                        </Card>
                    )}

                    <button
                        onClick={() => setPrediction(null)}
                        className="text-slate-400 hover:text-white hover:underline text-sm w-full text-center py-4 transition-colors"
                    >
                        Зробити новий аналіз
                    </button>
                </div>
            )}
        </div>
    );
};

export default Forecast;
