import React, { useState, useEffect, useMemo } from "react";
import {
    AreaChart,
    Area,
    BarChart,
    Bar,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    Legend,
} from "recharts";
import { ChevronLeft, ChevronRight, Calendar as CalendarIcon, Activity, AlertTriangle } from "lucide-react";
import api from "../api/axiosConfig.js";
import { Card } from "../components/UI.jsx";
import {
    format,
    startOfWeek,
    endOfWeek,
    eachDayOfInterval,
    isSameDay,
    subDays,
    addDays,
    subWeeks,
    addWeeks,
    subMonths,
    addMonths,
    parseISO,
} from "date-fns";
import { uk } from "date-fns/locale";
import styles from "../module_styles/Overview.module.css"; // Імпортуємо стилі

const Overview = () => {
    const [stats, setStats] = useState([]);
    const [timeRange, setTimeRange] = useState("week");
    const [currentDate, setCurrentDate] = useState(new Date());
    const tooltipStyles = {
        backgroundColor: "#0f172a",
        borderColor: "#334155",
        color: "#fff",
        borderRadius: "8px",
        padding: "8px",
    };

    useEffect(() => {
        const loadStats = async () => {
            try {
                const res = await api.get("/api/stats/");
                setStats(res.data.sort((a, b) => new Date(a.date) - new Date(b.date)));
            } catch (e) {
                console.error("Load stats error", e);
            }
        };
        loadStats();
    }, []);

    // --- Data Filtering Logic (useMemo) ---
    const filteredData = useMemo(() => {
        if (!stats.length) return [];

        let start, end;

        // Logic to determine start/end of the view
        if (timeRange === "day") {
            start = currentDate;
            end = currentDate;
        } else if (timeRange === "week") {
            start = startOfWeek(currentDate, { weekStartsOn: 1 });
            end = endOfWeek(currentDate, { weekStartsOn: 1 });
        } else if (timeRange === "month") {
            // Використовуємо startOfWeek і endOfWeek, але для місяця потрібна інша логіка,
            // оскільки date-fns.startOfMonth/endOfMonth не імпортовані
            // Припустимо, що вони імпортовані коректно з попереднього коду.
            start = new Date(currentDate.getFullYear(), currentDate.getMonth(), 1);
            end = new Date(currentDate.getFullYear(), currentDate.getMonth() + 1, 0);
        }

        const daysInInterval = eachDayOfInterval({ start, end });

        return daysInInterval.map((day) => {
            const statForDay = stats.find((s) => isSameDay(parseISO(s.date), day));
            return {
                date: format(day, timeRange === "month" ? "dd.MM" : "EEE", { locale: uk }), // dd.MM для місяця, Пн/Вт/Ср для тижня
                fullDate: day,
                steps: statForDay?.steps || 0,
                resting_hr: statForDay?.resting_hr || null,
                minutesAsleep: statForDay?.minutesAsleep ? statForDay.minutesAsleep / 60 : 0, // Залишаємо число для графіка
                stress_score: statForDay?.stress_score || 0,
                hasData: !!statForDay,
            };
        });
    }, [stats, timeRange, currentDate]);

    // --- Navigation Handlers ---
    const handlePrev = () => {
        if (timeRange === "day") setCurrentDate(subDays(currentDate, 1));
        if (timeRange === "week") setCurrentDate(subWeeks(currentDate, 1));
        if (timeRange === "month") setCurrentDate(subMonths(currentDate, 1));
    };

    const handleNext = () => {
        if (timeRange === "day") setCurrentDate(addDays(currentDate, 1));
        if (timeRange === "week") setCurrentDate(addWeeks(currentDate, 1));
        if (timeRange === "month") setCurrentDate(addMonths(currentDate, 1));
    };

    // --- Header Text ---
    const headerLabel = useMemo(() => {
        if (timeRange === "day") return format(currentDate, "d MMMM yyyy", { locale: uk });
        if (timeRange === "week") {
            const start = startOfWeek(currentDate, { weekStartsOn: 1 });
            const end = endOfWeek(currentDate, { weekStartsOn: 1 });
            return `${format(start, "d MMM", { locale: uk })} - ${format(end, "d MMM", { locale: uk })}`;
        }
        if (timeRange === "month") return format(currentDate, "LLLL yyyy", { locale: uk });
    }, [timeRange, currentDate]);

    // --- Render Content based on View ---
    const renderContent = () => {
        const hasAnyData = filteredData.some((d) => d.hasData);
        const latestDayStat =
            filteredData.find((d) => isSameDay(d.fullDate, currentDate)) || filteredData.find((d) => d.hasData);

        if (!stats.length) {
            return (
                <div className={styles.emptyState}>
                    <Activity className={styles.emptyIcon} />
                    <p className={styles.emptyText}>Завантаження даних...</p>
                </div>
            );
        }

        // --- RENDER DAY VIEW ---
        if (timeRange === "day") {
            if (!latestDayStat || !latestDayStat.hasData) {
                return (
                    <div className={styles.emptyState}>
                        <AlertTriangle className={styles.emptyIcon} />
                        <p className={styles.emptyText}>Немає даних за {headerLabel}. Додайте запис вручну.</p>
                    </div>
                );
            }

            return (
                <div className={styles.kpiGrid}>
                    {/* Використовуємо DayStat для відображення детальних KPI */}
                    {[
                        { label: "Пульс (RHR)", val: latestDayStat.resting_hr || "--", unit: "bpm", color: "blue" },
                        { label: "Кроки", val: latestDayStat.steps || "--", unit: "", color: "green" },
                        {
                            label: "Сон",
                            val: latestDayStat.minutesAsleep ? latestDayStat.minutesAsleep.toFixed(1) : "--",
                            unit: "год",
                            color: "purple",
                        },
                        { label: "Стрес", val: latestDayStat.stress_score || "--", unit: "/100", color: "orange" },
                    ].map((k, i) => (
                        <Card key={i} className={styles[`kpiCard${k.color}`]}>
                            <div className={styles.kpiLabel}>{k.label}</div>
                            <div className={styles.kpiValue}>
                                {k.val} <span className={styles.kpiUnit}>{k.unit}</span>
                            </div>
                        </Card>
                    ))}
                </div>
            );
        }

        // --- RENDER WEEK & MONTH VIEW (Charts) ---
        if (timeRange === "week" || timeRange === "month") {
            if (!hasAnyData) {
                return (
                    <div className={styles.emptyState}>
                        <AlertTriangle className={styles.emptyIcon} />
                        <p className={styles.emptyText}>Немає даних у цьому періоді. Спробуйте інший тиждень/місяць.</p>
                    </div>
                );
            }

            return (
                <div className={styles.chartContainerWrapper}>
                    {/* 1. Steps Chart (Bar) */}
                    <Card className={styles.chartCard}>
                        <h3 className={styles.chartTitle}>
                            <div className={styles.titleIndicatorGreen}></div>
                            Активність (Кроки)
                        </h3>
                        <ResponsiveContainer width="100%" height="85%">
                            <BarChart data={filteredData}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                                <XAxis
                                    dataKey="date"
                                    stroke="#64748b"
                                    tick={{ fontSize: 12 }}
                                    interval={timeRange === "month" ? "preserveStartEnd" : 0}
                                />
                                <YAxis stroke="#64748b" tick={{ fontSize: 12 }} />
                                <Tooltip contentStyle={tooltipStyles} cursor={{ fill: "rgba(30, 41, 59, 0.5)" }} />
                                <Bar dataKey="steps" fill="#22c55e" radius={[4, 4, 0, 0]} name="Кроки" />
                            </BarChart>
                        </ResponsiveContainer>
                    </Card>

                    <div className={styles.grid2Col}>
                        {/* 2. Heart Rate Chart (Area) */}
                        <Card className={styles.chartCardSmall}>
                            <h3 className={styles.chartTitleSmall}>
                                <div className={styles.titleIndicatorBlue}></div>
                                Пульс (RHR)
                            </h3>
                            <ResponsiveContainer width="100%" height="85%">
                                <AreaChart data={filteredData}>
                                    <defs>
                                        <linearGradient id="colorHr" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                                            <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                                        </linearGradient>
                                    </defs>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                                    <XAxis dataKey="date" stroke="#64748b" tick={{ fontSize: 10 }} />
                                    <YAxis
                                        domain={["dataMin - 5", "dataMax + 5"]}
                                        stroke="#64748b"
                                        tick={{ fontSize: 10 }}
                                    />
                                    <Tooltip contentStyle={tooltipStyles} />
                                    <Area
                                        type="monotone"
                                        dataKey="resting_hr"
                                        stroke="#3b82f6"
                                        strokeWidth={2}
                                        fill="url(#colorHr)"
                                        name="Пульс"
                                        connectNulls
                                    />
                                </AreaChart>
                            </ResponsiveContainer>
                        </Card>

                        {/* 3. Sleep Chart (Area/Line) */}
                        <Card className={styles.chartCardSmall}>
                            <h3 className={styles.chartTitleSmall}>
                                <div className={styles.titleIndicatorPurple}></div>
                                Тривалість сну
                            </h3>
                            <ResponsiveContainer width="100%" height="85%">
                                <AreaChart data={filteredData}>
                                    <defs>
                                        <linearGradient id="colorSleep" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="5%" stopColor="#a855f7" stopOpacity={0.3} />
                                            <stop offset="95%" stopColor="#a855f7" stopOpacity={0} />
                                        </linearGradient>
                                    </defs>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                                    <XAxis dataKey="date" stroke="#64748b" tick={{ fontSize: 10 }} />
                                    <YAxis stroke="#64748b" tick={{ fontSize: 10 }} />
                                    <Tooltip contentStyle={tooltipStyles} />
                                    <Area
                                        type="monotone"
                                        dataKey="minutesAsleep"
                                        stroke="#a855f7"
                                        strokeWidth={2}
                                        fill="url(#colorSleep)"
                                        name="Годин"
                                        connectNulls
                                    />
                                </AreaChart>
                            </ResponsiveContainer>
                        </Card>
                    </div>
                </div>
            );
        }
    };

    return (
        <div className={styles.pageWrapper}>
            {/* --- CONTROLS HEADER --- */}
            <div className={styles.controlsHeader}>
                {/* View Switcher */}
                <div className={styles.rangeSwitcher}>
                    {["day", "week", "month"].map((t) => (
                        <button
                            key={t}
                            onClick={() => setTimeRange(t)}
                            className={`${styles.rangeBtn} ${
                                timeRange === t ? styles.rangeBtnActive : styles.rangeBtnInactive
                            }`}
                        >
                            {t === "day" && "День"}
                            {t === "week" && "Тиждень"}
                            {t === "month" && "Місяць"}
                        </button>
                    ))}
                </div>

                {/* Date Navigation */}
                <div className={styles.dateNavigator}>
                    <button onClick={handlePrev} className={styles.navBtn}>
                        <ChevronLeft size={20} />
                    </button>

                    <div className={styles.dateLabel}>
                        <CalendarIcon size={16} className={styles.calendarIcon} />
                        {headerLabel}
                    </div>

                    <button onClick={handleNext} className={styles.navBtn}>
                        <ChevronRight size={20} />
                    </button>
                </div>
            </div>

            {/* --- MAIN CONTENT --- */}
            {renderContent()}
        </div>
    );
};

export default Overview;
