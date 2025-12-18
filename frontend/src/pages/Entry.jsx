import React, { useState, useEffect, useMemo } from "react";
import { Calendar as CalendarIcon, Save, RefreshCw, Check, ChevronLeft, ChevronRight } from "lucide-react";
import { useStats } from "../context/StatsContext.jsx";
import { Card, Input } from "../components/UI.jsx";
import {
    format,
    isSameDay,
    parseISO,
    startOfMonth,
    endOfMonth,
    eachDayOfInterval,
    startOfWeek,
    endOfWeek,
    isToday,
    addMonths,
    subMonths,
} from "date-fns";
import { uk } from "date-fns/locale";
import styles from "../module_styles/Entry.module.css";

const Entry = () => {
    const { stats, fetchStats, addStat, updateStat } = useStats();

    const [entry, setEntry] = useState({
        date: format(new Date(), "yyyy-MM-dd"),
        steps: 0,
        minutesAsleep: 0,
        stress_score: 0,
        nightly_temperature: 36.6,
        resting_hr: 0,
        very_active_minutes: 0,
        sleep_efficiency: 0,
    });

    const [loading, setLoading] = useState(false);
    const [mode, setMode] = useState("create");
    const [viewDate, setViewDate] = useState(new Date());

    const handleChange = (name, value) => {
        setEntry((prev) => ({ ...prev, [name]: value }));
    };

    useEffect(() => {
        fetchStats();
    }, [fetchStats]);

    // Цей useEffect тепер автоматично спрацює, коли StatsContext оновиться
    useEffect(() => {
        const existingStat = stats.find((s) => s.date === entry.date);
        if (existingStat) {
            // Важливо: копіюємо дані, щоб не мутувати стейт напряму
            setEntry({ ...existingStat });
            setMode("update");
        } else {
            setEntry((prev) => ({
                ...prev,
                steps: 0,
                minutesAsleep: 0,
                stress_score: 0,
                resting_hr: 0,
                very_active_minutes: 0,
                sleep_efficiency: 0,
            }));
            setMode("create");
        }
    }, [entry.date, stats]); // stats тут — ключ до динаміки

    const handleSave = async () => {
        setLoading(true);
        try {
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

            if (mode === "update") {
                await updateStat(entry.date, payload);
            } else {
                await addStat(payload);
                // Після створення режим автоматично зміниться на 'update'
                // через useEffect вище, бо запис з'явиться в stats
            }

            // Замість alert можна використовувати тости (toast), це приємніше
            console.log("Дані синхронізовано з хмарою");
        } catch (e) {
            alert("Помилка: " + (e.response?.data?.detail || e.message));
        } finally {
            setLoading(false);
        }
    };

    const calendarDays = useMemo(() => {
        const start = startOfWeek(startOfMonth(viewDate), { weekStartsOn: 1 });
        const end = endOfWeek(endOfMonth(viewDate), { weekStartsOn: 1 });
        return eachDayOfInterval({ start, end });
    }, [viewDate]);

    return (
        <div className={styles.container}>
            <div className={styles.layoutGrid}>
                {/* КАЛЕНДАР */}
                <Card className={styles.calendarCard}>
                    <div className={styles.calendarHeader}>
                        <button onClick={() => setViewDate(subMonths(viewDate, 1))}>
                            <ChevronLeft />
                        </button>
                        <h4 className="capitalize">{format(viewDate, "LLLL yyyy", { locale: uk })}</h4>
                        <button onClick={() => setViewDate(addMonths(viewDate, 1))}>
                            <ChevronRight />
                        </button>
                    </div>

                    <div className={styles.calendarGrid}>
                        {["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Нд"].map((d) => (
                            <div key={d} className={styles.weekdayLabel}>
                                {d}
                            </div>
                        ))}
                        {calendarDays.map((day) => {
                            const dateStr = format(day, "yyyy-MM-dd");
                            // Динамічна перевірка: якщо дата є в масиві stats — кнопка стане зеленою
                            const isFilled = stats.some((s) => s.date === dateStr);
                            const isSelected = entry.date === dateStr;

                            return (
                                <button
                                    key={dateStr}
                                    onClick={() => handleChange("date", dateStr)}
                                    className={`
                                        ${styles.dayBtn} 
                                        ${isFilled ? styles.dayFilled : styles.dayEmpty}
                                        ${isSelected ? styles.daySelected : ""}
                                        ${
                                            !isSameDay(day, viewDate) && format(day, "M") !== format(viewDate, "M")
                                                ? styles.dayOffMonth
                                                : ""
                                        }
                                    `}
                                >
                                    {format(day, "d")}
                                    {isToday(day) && <div className={styles.todayIndicator} />}
                                </button>
                            );
                        })}
                    </div>
                    <div className={styles.legend}>
                        <div className={styles.legendItem}>
                            <span className={styles.dotGreen} /> Заповнено
                        </div>

                        <div className={styles.legendItem}>
                            <span className={styles.dotRed} /> Порожньо
                        </div>
                    </div>
                </Card>

                {/* ФОРМА */}
                <Card className={styles.formCard}>
                    <div className={styles.formHeader}>
                        <h3 className={styles.title}>{mode === "update" ? "Дані зафіксовано" : "Новий запис"}</h3>
                        <div className={styles.dateBadge}>
                            <CalendarIcon size={14} />
                            {format(parseISO(entry.date), "dd MMMM", { locale: uk })}
                        </div>
                    </div>

                    <div className={styles.inputsGrid}>
                        {/* Твої Input компоненти тут */}
                        <Input
                            label="Пульс (RHR)"
                            type="number"
                            value={entry.resting_hr}
                            onChange={(v) => handleChange("resting_hr", v)}
                        />
                        <Input
                            label="Кроки"
                            type="number"
                            value={entry.steps}
                            onChange={(v) => handleChange("steps", v)}
                        />

                        <Input
                            label="Сон (хв)"
                            type="number"
                            value={entry.minutesAsleep}
                            onChange={(v) => handleChange("minutesAsleep", v)}
                        />

                        <Input
                            label="Стрес"
                            type="number"
                            value={entry.stress_score}
                            onChange={(v) => handleChange("stress_score", v)}
                        />

                        <Input
                            label="Активність (хв)"
                            type="number"
                            value={entry.very_active_minutes}
                            onChange={(v) => handleChange("very_active_minutes", v)}
                        />

                        <Input
                            label="Температура"
                            type="number"
                            value={entry.nightly_temperature}
                            onChange={(v) => handleChange("nightly_temperature", v)}
                        />

                        <Input
                            label="Ефективність сну (%)"
                            type="number"
                            value={entry.sleep_efficiency}
                            onChange={(v) => handleChange("sleep_efficiency", v)}
                        />
                    </div>

                    <button
                        onClick={handleSave}
                        disabled={loading}
                        className={mode === "update" ? styles.updateBtn : styles.saveBtn}
                    >
                        {loading ? <RefreshCw className="animate-spin" /> : mode === "update" ? <Check /> : <Save />}
                        {mode === "update" ? "Оновити дані" : "Зберегти запис"}
                    </button>
                </Card>
            </div>
        </div>
    );
};

export default Entry;
