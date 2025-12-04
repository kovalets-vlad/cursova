import React, { useState, useEffect, createContext, useContext } from "react";
import api from "../api/axiosConfig.js";

const AuthContext = createContext(null);

export const useAuth = () => {
    const context = useContext(AuthContext);
    if (context === null) {
        throw new Error("useAuth must be used within an AuthProvider");
    }
    return context;
};

export const AuthProvider = ({ children }) => {
    const [user, setUser] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const token = localStorage.getItem("token");
        const savedUser = localStorage.getItem("user");
        if (token && savedUser) setUser(JSON.parse(savedUser));
        setLoading(false);
    }, []);

    const login = async (username, password) => {
        try {
            // Виправлено: надсилаємо JSON-тіло
            const res = await api.post("/api/auth/login", {
                username,
                password,
            });

            localStorage.setItem("token", res.data.access_token);
            localStorage.setItem("refresh_token", res.data.refresh_token);

            const userData = { username };
            localStorage.setItem("user", JSON.stringify(userData));
            setUser(userData);
            return { success: true };
        } catch (e) {
            console.error("Login error:", e);
            return { success: false, error: e.response?.data?.detail || "Помилка входу" };
        }
    };

    const register = async (username, password, email) => {
        try {
            await api.post("/api/auth/register", { username, password, email });
            await login(username, password);
            return { success: true };
        } catch (e) {
            console.error("Register error:", e);
            return { success: false, error: e.response?.data?.detail || "Помилка реєстрації" };
        }
    };

    const logout = () => {
        localStorage.removeItem("token");
        localStorage.removeItem("refresh_token");
        localStorage.removeItem("user");
        setUser(null);
    };

    return (
        <AuthContext.Provider value={{ user, login, register, logout, loading }}>
            {!loading && children}
        </AuthContext.Provider>
    );
};
