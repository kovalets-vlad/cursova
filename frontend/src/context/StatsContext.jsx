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

    //     // --- ДОДАВАННЯ ЗАПИСУ ---
    const addStat = async (newStat) => {
        try {
            const response = await api.post("/api/stats/manual_entry", newStat);

            const createdStat = response.data.data;

            setStats((prev) => [...prev, createdStat]);
            return createdStat;
        } catch (error) {
            console.error("Помилка при створенні:", error);
            throw error;
        }
    };

    // --- ОНОВЛЕННЯ ЗАПИСУ ---
    const updateStat = async (date, updatedData) => {
        try {
            const response = await api.patch(`/api/stats/${date}`, updatedData);

            const updatedStat = response.data;

            setStats((prev) => prev.map((s) => (s.date === date ? updatedStat : s)));
            return updatedStat;
        } catch (error) {
            console.error("Помилка при оновленні:", error);
            throw error;
        }
    };

    return (
        <StatsContext.Provider value={{ stats, loading, isLoaded, fetchStats, addStat, updateStat }}>
            {children}
        </StatsContext.Provider>
    );
};

export const useStats = () => useContext(StatsContext);
