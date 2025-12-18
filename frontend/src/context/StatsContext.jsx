import React, { createContext, useContext, useState, useCallback } from "react";
import api from "../api/axiosConfig.js";

const StatsContext = createContext();

export const StatsProvider = ({ children }) => {
    const [stats, setStats] = useState([]);
    const [isLoaded, setIsLoaded] = useState(false);
    const [loading, setLoading] = useState(false);

    // Початкове завантаження
    const fetchStats = useCallback(
        async (force = false) => {
            if (isLoaded && !force) return;
            setLoading(true);
            try {
                const res = await api.get("/api/stats/");
                setStats(res.data);
                setIsLoaded(true);
            } catch (e) {
                console.error("Помилка завантаження статистики", e);
            } finally {
                setLoading(false);
            }
        },
        [isLoaded]
    );

    // Додавання (Локально + Бекенд)
    const addStat = async (newEntry) => {
        const res = await api.post("/api/stats/manual_entry", newEntry);
        const savedEntry = res.data;
        // Оновлюємо стейт: додаємо новий запис до масиву
        setStats((prev) => [...prev, savedEntry]);
        return savedEntry;
    };

    // Оновлення (Локально + Бекенд)
    const updateStat = async (date, updatedData) => {
        const res = await api.patch(`/api/stats/${date}`, updatedData);
        const savedEntry = res.data;
        // Оновлюємо стейт: замінюємо старий об'єкт новим за датою
        setStats((prev) => prev.map((s) => (s.date === date ? savedEntry : s)));
        return savedEntry;
    };

    return (
        <StatsContext.Provider value={{ stats, loading, isLoaded, fetchStats, addStat, updateStat }}>
            {children}
        </StatsContext.Provider>
    );
};

export const useStats = () => useContext(StatsContext);
