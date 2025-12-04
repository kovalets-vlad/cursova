import React, { useState, useEffect } from "react";
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import { Activity, Heart, Brain, User, LogOut, PlusCircle, TrendingUp, Mail, Lock } from "lucide-react";
import api from "../api/axiosConfig.js";
import { useAuth } from "../context/AuthContext.jsx";
import { Card, Input } from "../components/UI.jsx";

const Dashboard = () => {
    const { logout, user } = useAuth();
    const [tab, setTab] = useState("overview");
    const [stats, setStats] = useState([]);
    const [loading, setLoading] = useState(false);
    const [prediction, setPrediction] = useState(null);
    const [aiAdvice, setAiAdvice] = useState(null);
    const [notification, setNotification] = useState(null); // Для повідомлень

    // Форма для введення
    const [entry, setEntry] = useState({
        date: new Date().toISOString().split("T")[0],
        steps: 8000,
        minutesAsleep: 420,
        stress_score: 30,
        nightly_temperature: 36.6,
        resting_hr: 60,
        very_active_minutes: 30,
        sleep_efficiency: 90,
    });

    // Хелпер для показу повідомлень
    const showNotification = (message, isError = false) => {
        setNotification({ message, isError });
        setTimeout(() => setNotification(null), 5000);
    };

    // Завантаження даних
    const loadStats = async () => {
        try {
            const res = await api.get("/api/stats/");
            // Обробляємо дані: сортуємо від старого до нового для графіка
            setStats(res.data.sort((a, b) => new Date(a.date) - new Date(b.date)));
        } catch (e) {
            showNotification("Не вдалося завантажити статистику.", true);
            console.error("Load stats error:", e);
        }
    };

    useEffect(() => {
        loadStats();
    }, []);

    const handleChangeEntry = (name, value) => {
        setEntry((prev) => ({ ...prev, [name]: value }));
    };

    const handleAddEntry = async () => {
        try {
            // Перетворення числових рядків у числа, якщо Input не робить цього автоматично
            const payload = {
                ...entry,
                steps: Number(entry.steps),
                minutesAsleep: Number(entry.minutesAsleep),
                stress_score: Number(entry.stress_score),
                nightly_temperature: Number(entry.nightly_temperature),
                resting_hr: Number(entry.resting_hr),
                very_active_minutes: Number(entry.very_active_minutes),
                sleep_efficiency: Number(entry.sleep_efficiency),
            };

            await api.post("/api/stats/manual_entry", payload);
            showNotification("Запис успішно додано!", false);
            loadStats();
            setTab("overview");
        } catch (e) {
            showNotification("Помилка додавання: " + (e.response?.data?.detail || e.message), true);
            console.error("Add entry error:", e);
        }
    };

    const handlePredict = async () => {
        setLoading(true);
        setPrediction(null);
        setAiAdvice(null);
        try {
            // 1. Отримуємо прогноз (цифри)
            const res = await api.post("/api/predict_pulse");
            setPrediction(res.data);

            // 2. Отримуємо пораду від ШІ
            const lastStat = stats[stats.length - 1] || {};

            const adviceRes = await api.post("/api/prediction/advice", {
                user_stats: {
                    age: 30, // Можна взяти з профілю
                    stress_score: lastStat.stress_score || 50,
                    minutesAsleep: lastStat.minutesAsleep || 400,
                    sleep_efficiency: lastStat.sleep_efficiency || 85,
                    steps: lastStat.steps || 5000,
                    acwr: res.data.acwr_from_backend || 1.0, // Краще, якщо бекенд сам його розрахує і поверне
                },
                prediction_delta: res.data.predicted_delta,
                predicted_bpm: res.data.predicted_bpm,
            });
            setAiAdvice(adviceRes.data.advice);
            showNotification("Прогноз успішно згенеровано.", false);
        } catch (e) {
            showNotification("Помилка прогнозу. Можливо, недостатньо даних.", true);
            console.error("Prediction error:", e.response?.data?.detail || e.message);
        } finally {
            setLoading(false);
        }
    };

    const lastStat = stats[stats.length - 1];
    const dataForChart = stats.map((s) => ({
        date: s.date.substring(5), // Показуємо тільки місяць і день
        resting_hr: s.resting_hr,
    }));

    return (
        <div className="min-h-screen bg-slate-950 text-slate-200 font-sans flex">
            {/* Notification Bar */}
            {notification && (
                <div
                    className={`fixed top-4 right-4 z-50 p-4 rounded-xl shadow-2xl transition-all duration-300 ${
                        notification.isError ? "bg-red-600" : "bg-green-600"
                    }`}
                >
                    {notification.message}
                </div>
            )}

            {/* SIDEBAR */}
            <aside className="w-64 bg-slate-900 border-r border-slate-800 p-6 hidden md:flex flex-col">
                <div className="flex items-center gap-3 text-blue-500 mb-10">
                    <Activity className="w-8 h-8" />
                    <span className="text-2xl font-bold text-white">Pulse AI</span>
                </div>

                <nav className="flex-1 space-y-2">
                    {[
                        { id: "overview", icon: TrendingUp, label: "Огляд" },
                        { id: "forecast", icon: Brain, label: "AI Прогноз" },
                        { id: "entry", icon: PlusCircle, label: "Додати запис" },
                        { id: "profile", icon: User, label: "Профіль" },
                    ].map((item) => (
                        <button
                            key={item.id}
                            onClick={() => setTab(item.id)}
                            className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all ${
                                tab === item.id
                                    ? "bg-blue-600 text-white shadow-lg shadow-blue-900/20"
                                    : "text-slate-400 hover:bg-slate-800 hover:text-white"
                            }`}
                        >
                            <item.icon className="w-5 h-5" />
                            <span className="font-medium">{item.label}</span>
                        </button>
                    ))}
                </nav>

                <button
                    onClick={logout}
                    className="flex items-center gap-3 px-4 py-3 text-slate-400 hover:text-red-400 hover:bg-red-500/10 rounded-xl transition-all"
                >
                    <LogOut className="w-5 h-5" />
                    <span>Вихід</span>
                </button>
            </aside>

            {/* MAIN CONTENT */}
            <main className="flex-1 p-4 md:p-8 overflow-y-auto h-screen">
                <header className="flex justify-between items-center mb-8">
                    <h2 className="text-xl md:text-3xl font-bold text-white">
                        {tab === "overview" && "Головна панель"}
                        {tab === "forecast" && "Прогноз здоров'я"}
                        {tab === "entry" && "Введення даних"}
                        {tab === "profile" && "Налаштування профілю"}
                    </h2>
                    <div className="flex items-center gap-3 bg-slate-900 px-4 py-2 rounded-full border border-slate-800">
                        <div className="w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center text-white font-bold">
                            {user?.username?.[0].toUpperCase()}
                        </div>
                        <span className="text-slate-300 hidden sm:inline">{user?.username}</span>
                    </div>
                </header>

                {/* --- OVERVIEW TAB --- */}
                {tab === "overview" && (
                    <div className="space-y-6">
                        {/* KPI Cards */}
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                            {[
                                {
                                    label: "Пульс (RHR)",
                                    val: lastStat?.resting_hr || "--",
                                    unit: "bpm",
                                    color: "text-blue-400",
                                },
                                {
                                    label: "Кроки",
                                    val: lastStat?.steps || "--",
                                    unit: "",
                                    color: "text-green-400",
                                },
                                {
                                    label: "Сон",
                                    val: lastStat ? (lastStat.minutesAsleep / 60).toFixed(1) : "--",
                                    unit: "год",
                                    color: "text-purple-400",
                                },
                                {
                                    label: "Стрес",
                                    val: lastStat?.stress_score || "--",
                                    unit: "/100",
                                    color: "text-orange-400",
                                },
                            ].map((k, i) => (
                                <Card key={i}>
                                    <div className="text-slate-400 text-sm mb-2">{k.label}</div>
                                    <div className={`text-2xl md:text-3xl font-bold ${k.color}`}>
                                        {k.val} <span className="text-sm text-slate-500">{k.unit}</span>
                                    </div>
                                </Card>
                            ))}
                        </div>

                        {/* Chart */}
                        <Card className="h-72 md:h-96">
                            <h3 className="text-xl font-semibold text-white mb-6">
                                Динаміка пульсу (останні {stats.length} днів)
                            </h3>
                            <ResponsiveContainer width="100%" height="80%">
                                <AreaChart data={dataForChart} margin={{ top: 10, right: 30, left: -20, bottom: 0 }}>
                                    <defs>
                                        <linearGradient id="colorHr" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                                            <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                                        </linearGradient>
                                    </defs>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                                    <XAxis dataKey="date" stroke="#64748b" />
                                    <YAxis domain={["dataMin - 5", "dataMax + 5"]} stroke="#64748b" />
                                    <Tooltip contentStyle={{ backgroundColor: "#0f172a", borderColor: "#334155" }} />
                                    <Area
                                        type="monotone"
                                        dataKey="resting_hr"
                                        stroke="#3b82f6"
                                        fillOpacity={1}
                                        fill="url(#colorHr)"
                                        name="Пульс спокою (bpm)"
                                    />
                                </AreaChart>
                            </ResponsiveContainer>
                        </Card>
                    </div>
                )}

                {/* --- FORECAST TAB (скорочено для читабельності) --- */}
                {tab === "forecast" && (
                    <div className="max-w-4xl mx-auto space-y-8">
                        <div className="text-center py-10">
                            {!prediction ? (
                                <>
                                    <div className="w-24 h-24 bg-slate-800 rounded-full flex items-center justify-center mx-auto mb-6 border border-slate-700">
                                        <Brain className="w-12 h-12 text-blue-500" />
                                    </div>
                                    <h3 className="text-2xl text-white font-bold mb-2">ШІ Аналіз Відновлення</h3>
                                    <p className="text-slate-400 mb-8 max-w-lg mx-auto">
                                        Нейромережа проаналізує твої показники за останні 7 днів та спрогнозує стан на
                                        завтра. (Потрібно мінімум 5 записів)
                                    </p>
                                    <button
                                        onClick={handlePredict}
                                        disabled={loading || stats.length < 5}
                                        className="bg-blue-600 hover:bg-blue-700 text-white px-8 py-3 rounded-xl font-bold text-lg shadow-lg shadow-blue-600/20 transition-all disabled:opacity-50"
                                    >
                                        {loading ? "Аналізую..." : "Згенерувати прогноз"}
                                    </button>
                                    {stats.length < 5 && (
                                        <p className="mt-4 text-red-400 text-sm">
                                            Потрібно мінімум 5 днів даних для прогнозу.
                                        </p>
                                    )}
                                </>
                            ) : (
                                <div className="animate-fade-in space-y-6 text-left">
                                    <div className="grid md:grid-cols-2 gap-6">
                                        {/* Prediction Card */}
                                        <Card className="bg-gradient-to-br from-indigo-900/40 to-slate-900 border-indigo-500/30">
                                            <div className="flex items-center gap-2 mb-4 text-indigo-300">
                                                <Heart className="w-5 h-5" /> Прогноз пульсу
                                            </div>
                                            <div className="flex items-baseline gap-3">
                                                <span className="text-5xl font-bold text-white">
                                                    {prediction.predicted_bpm.toFixed(1)}
                                                </span>
                                                <span
                                                    className={`text-xl font-medium ${
                                                        prediction.predicted_delta > 0
                                                            ? "text-red-400"
                                                            : "text-green-400"
                                                    }`}
                                                >
                                                    {prediction.predicted_delta > 0 ? "↗" : "↘"}{" "}
                                                    {Math.abs(prediction.predicted_delta).toFixed(2)}
                                                </span>
                                            </div>
                                            <div className="mt-4 text-slate-400 text-sm">
                                                Поточний: {lastStat?.resting_hr} bpm
                                            </div>
                                        </Card>

                                        {/* Details Card */}
                                        <Card>
                                            <div className="text-sm text-slate-400 mb-4">Думка моделей (Delta):</div>
                                            <div className="space-y-3">
                                                {Object.entries(prediction.details || {}).map(([key, val]) => (
                                                    <div
                                                        key={key}
                                                        className="flex justify-between items-center p-2 bg-slate-900/50 rounded"
                                                    >
                                                        <span className="uppercase text-xs font-bold text-slate-500">
                                                            {key.split("_").join(" ").toUpperCase()}
                                                        </span>
                                                        <span
                                                            className={`font-mono ${
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

                                    {/* AI Advice */}
                                    {aiAdvice && (
                                        <Card className="border-blue-500/30 bg-blue-900/10">
                                            <div className="flex items-center gap-3 mb-4">
                                                <div className="p-2 bg-blue-500 rounded-lg text-white">
                                                    <Brain className="w-5 h-5" />
                                                </div>
                                                <h3 className="text-xl font-bold text-white">Рекомендація Тренера</h3>
                                            </div>
                                            <div className="prose prose-invert max-w-none text-slate-200 whitespace-pre-line leading-relaxed">
                                                {aiAdvice}
                                            </div>
                                        </Card>
                                    )}

                                    <button
                                        onClick={() => setPrediction(null)}
                                        className="text-slate-400 hover:text-white underline text-sm w-full text-center"
                                    >
                                        Зробити новий аналіз
                                    </button>
                                </div>
                            )}
                        </div>
                    </div>
                )}

                {/* --- ENTRY TAB --- */}
                {tab === "entry" && (
                    <div className="max-w-2xl mx-auto">
                        <Card>
                            <h3 className="text-xl font-bold text-white mb-6">Додати дані за день</h3>
                            <div className="grid grid-cols-2 gap-4">
                                <Input
                                    label="Дата"
                                    type="date"
                                    value={entry.date}
                                    onChange={(v) => handleChangeEntry("date", v)}
                                />
                                <Input
                                    label="Кроки"
                                    type="number"
                                    value={entry.steps}
                                    onChange={(v) => handleChangeEntry("steps", v)}
                                />
                                <Input
                                    label="Активні хвилини"
                                    type="number"
                                    value={entry.very_active_minutes}
                                    onChange={(v) => handleChangeEntry("very_active_minutes", v)}
                                />
                                <Input
                                    label="Сон (хв)"
                                    type="number"
                                    value={entry.minutesAsleep}
                                    onChange={(v) => handleChangeEntry("minutesAsleep", v)}
                                />
                                <Input
                                    label="Ефективність сну (%)"
                                    type="number"
                                    value={entry.sleep_efficiency}
                                    onChange={(v) => handleChangeEntry("sleep_efficiency", v)}
                                />
                                <Input
                                    label="Стрес (0-100)"
                                    type="number"
                                    value={entry.stress_score}
                                    onChange={(v) => handleChangeEntry("stress_score", v)}
                                />
                                <Input
                                    label="Температура"
                                    type="number"
                                    value={entry.nightly_temperature}
                                    onChange={(v) => handleChangeEntry("nightly_temperature", v)}
                                />
                                <Input
                                    label="Пульс спокою"
                                    type="number"
                                    value={entry.resting_hr}
                                    onChange={(v) => handleChangeEntry("resting_hr", v)}
                                />
                            </div>
                            <button
                                onClick={handleAddEntry}
                                className="w-full mt-6 bg-green-600 hover:bg-green-700 text-white py-3 rounded-xl font-bold shadow-lg shadow-green-900/20 transition-all"
                            >
                                Зберегти в базу
                            </button>
                        </Card>
                    </div>
                )}
                {/* --- PROFILE TAB --- */}
                {tab === "profile" && (
                    <div className="max-w-xl mx-auto">
                        <Card>
                            <h3 className="text-xl font-bold text-white mb-6">Дані користувача</h3>
                            <div className="space-y-4 text-slate-300">
                                <div className="flex items-center gap-3">
                                    <User className="w-5 h-5 text-blue-400" />
                                    <span className="font-medium">Логін: {user?.username}</span>
                                </div>
                                <div className="flex items-center gap-3">
                                    <Mail className="w-5 h-5 text-blue-400" />
                                    <span className="font-medium">Email: {user?.email || "Не вказано"}</span>
                                </div>
                                <div className="flex items-center gap-3">
                                    <Lock className="w-5 h-5 text-blue-400" />
                                    <span className="font-medium">Пароль: ******</span>
                                </div>
                                <div className="flex items-center gap-3 pt-4">
                                    <button
                                        onClick={logout}
                                        className="flex items-center gap-2 text-red-400 hover:text-red-300 transition-all font-medium"
                                    >
                                        <LogOut className="w-5 h-5" />
                                        Вийти з акаунту
                                    </button>
                                </div>
                            </div>
                        </Card>
                    </div>
                )}
            </main>
        </div>
    );
};

export default Dashboard;
