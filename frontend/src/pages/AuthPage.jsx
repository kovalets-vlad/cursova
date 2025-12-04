import React, { useState } from "react";
import { Activity } from "lucide-react";
import { useAuth } from "../context/AuthContext.jsx";
import { Card, Input } from "../components/UI.jsx";

const AuthPage = () => {
    const { login, register } = useAuth();
    const [isLogin, setIsLogin] = useState(true);
    const [formData, setFormData] = useState({ username: "", password: "", email: "" });
    const [error, setError] = useState("");

    const handleChange = (name, value) => {
        setFormData((prev) => ({ ...prev, [name]: value }));
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError("");

        // Перевірка на пусті поля
        if (!formData.username || !formData.password || (!isLogin && !formData.email)) {
            setError("Будь ласка, заповніть усі поля.");
            return;
        }

        const res = isLogin
            ? await login(formData.username, formData.password)
            : await register(formData.username, formData.password, formData.email);

        if (!res.success) setError(res.error);
    };

    return (
        <div className="min-h-screen flex items-center justify-center bg-slate-950 p-4">
            <Card className="w-full max-w-md border-slate-800 bg-slate-900/80">
                <div className="text-center mb-8">
                    <div className="inline-flex p-3 bg-blue-600 rounded-xl mb-4 shadow-lg shadow-blue-500/20">
                        <Activity className="w-8 h-8 text-white" />
                    </div>
                    <h1 className="text-2xl font-bold text-white">Pulse AI</h1>
                    <p className="text-slate-400">Твій розумний кардіо-асистент</p>
                </div>

                <div className="flex bg-slate-800 rounded-lg p-1 mb-6">
                    <button
                        onClick={() => {
                            setIsLogin(true);
                            setError("");
                        }}
                        className={`flex-1 py-2 rounded-md text-sm font-medium transition-all ${
                            isLogin ? "bg-slate-700 text-white shadow" : "text-slate-400"
                        }`}
                    >
                        Вхід
                    </button>
                    <button
                        onClick={() => {
                            setIsLogin(false);
                            setError("");
                        }}
                        className={`flex-1 py-2 rounded-md text-sm font-medium transition-all ${
                            !isLogin ? "bg-slate-700 text-white shadow" : "text-slate-400"
                        }`}
                    >
                        Реєстрація
                    </button>
                </div>

                <form onSubmit={handleSubmit} className="space-y-4">
                    <Input label="Логін" value={formData.username} onChange={(v) => handleChange("username", v)} />
                    {!isLogin && (
                        <Input
                            label="Email"
                            type="email"
                            value={formData.email}
                            onChange={(v) => handleChange("email", v)}
                        />
                    )}
                    <Input
                        label="Пароль"
                        type="password"
                        value={formData.password}
                        onChange={(v) => handleChange("password", v)}
                    />

                    {error && (
                        <div className="text-red-400 text-sm bg-red-500/10 p-3 rounded-lg border border-red-500/20">
                            {error}
                        </div>
                    )}

                    <button
                        type="submit"
                        className="w-full bg-blue-600 hover:bg-blue-700 text-white py-3 rounded-xl font-medium transition-all shadow-lg shadow-blue-500/25"
                    >
                        {isLogin ? "Увійти в систему" : "Створити акаунт"}
                    </button>
                </form>
            </Card>
        </div>
    );
};

export default AuthPage;
