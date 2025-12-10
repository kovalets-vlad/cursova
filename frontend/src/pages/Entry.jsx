import React, { useState } from "react";
import { Calendar, Save } from "lucide-react";
import api from "../api/axiosConfig.js";
import { Card, Input } from "../components/UI.jsx";
import { format, subDays, isSameDay } from "date-fns";
import styles from "../module_styles/Entry.module.css";
import { parseISO } from "date-fns";

// --- Хелпер для дат ---
const formatDate = (date) => format(date, "yyyy-MM-dd");

const Entry = () => {
    const initialEntryState = {
        date: formatDate(new Date()),
        steps: 8000,
        minutesAsleep: 420,
        stress_score: 30,
        nightly_temperature: 36.6,
        resting_hr: 60,
        very_active_minutes: 30,
        sleep_efficiency: 90,
    };

    const [entry, setEntry] = useState(initialEntryState);
    const [loading, setLoading] = useState(false);
    const [isCalendarOpen, setIsCalendarOpen] = useState(false);
    const handleChange = (name, value) => {
        setEntry((prev) => ({ ...prev, [name]: value }));
    };

    // Обробник для швидкого вибору дати
    const handleDateSelect = (date) => {
        setEntry((prev) => ({ ...prev, date: formatDate(date) }));
        setIsCalendarOpen(false);
    };

    const handleAddEntry = async () => {
        setLoading(true);
        try {
            const payload = {
                ...entry,
                // Забезпечуємо, що всі числові поля є числами
                ...Object.fromEntries(
                    Object.entries(entry).map(([key, value]) => [key, key === "date" ? value : Number(value)])
                ),
            };

            await api.post("/api/stats/manual_entry", payload);
            alert("Запис успішно додано!");
        } catch (e) {
            alert("Помилка: " + (e.response?.data?.detail || e.message));
        } finally {
            setLoading(false);
        }
    };

    // Елементи для швидкого вибору дати
    const dateOptions = [
        { label: "Сьогодні", date: new Date() },
        { label: "Вчора", date: subDays(new Date(), 1) },
        { label: "Позавчора", date: subDays(new Date(), 2) },
    ];

    return (
        <div className={styles.container}>
            <Card className={styles.cardWrapper}>
                <h3 className={styles.title}>Додати дані за день</h3>

                {/* --- 2. DATE PICKER --- */}
                <div className={styles.datePickerSection}>
                    <label className={styles.dateLabel}>Виберіть дату запису</label>
                    <div className={styles.dateControls}>
                        {/* Швидкий вибір (Сьогодні, Вчора, ...) */}
                        {dateOptions.map((opt) => (
                            <button
                                key={opt.label}
                                onClick={() => handleDateSelect(opt.date)}
                                className={`${styles.quickDateBtn} ${
                                    isSameDay(parseISO(entry.date), opt.date) ? styles.quickDateBtnActive : ""
                                }`}
                            >
                                {opt.label}
                            </button>
                        ))}

                        {/* Кнопка календаря */}
                        <button onClick={() => setIsCalendarOpen(true)} className={styles.calendarBtn}>
                            <Calendar size={18} />
                        </button>
                    </div>
                </div>

                {/* --- 3. РУЧНЕ ВВЕДЕННЯ --- */}
                <div className={styles.manualEntryGrid}>
                    {/* Якщо календар відкритий (isCalendarOpen), показуємо його замість Input[type=date] */}
                    {isCalendarOpen ? (
                        <div className={styles.customCalendarWrapper}>
                            {/* ТУТ БУДЕ КОМПОНЕНТ РЕАЛЬНОГО КАЛЕНДАРЯ (наприклад, React-Datepicker) */}
                            <p className={styles.calendarPlaceholder}>
                                *Тут має бути інтерактивний календар для вибору колись))))
                            </p>
                            <Input
                                label="Введіть дату"
                                type="date"
                                value={entry.date}
                                onChange={(v) => handleChange("date", v)}
                            />
                        </div>
                    ) : (
                        // Інакше Input[type=date] для відображення
                        <div className={styles.singleInputRow}>
                            <Input
                                label="Вибрана дата"
                                type="text"
                                value={format(parseISO(entry.date), "dd.MM.yyyy")}
                                disabled={true}
                            />
                        </div>
                    )}

                    <Input
                        label="Пульс спокою (bpm)"
                        type="number"
                        value={entry.resting_hr}
                        onChange={(v) => handleChange("resting_hr", v)}
                    />
                    <Input label="Кроки" type="number" value={entry.steps} onChange={(v) => handleChange("steps", v)} />
                    <Input
                        label="Активні хвилини"
                        type="number"
                        value={entry.very_active_minutes}
                        onChange={(v) => handleChange("very_active_minutes", v)}
                    />

                    <Input
                        label="Сон (хв)"
                        type="number"
                        value={entry.minutesAsleep}
                        onChange={(v) => handleChange("minutesAsleep", v)}
                    />
                    <Input
                        label="Ефективність сну (%)"
                        type="number"
                        value={entry.sleep_efficiency}
                        onChange={(v) => handleChange("sleep_efficiency", v)}
                    />
                    <Input
                        label="Стрес (0-100)"
                        type="number"
                        value={entry.stress_score}
                        onChange={(v) => handleChange("stress_score", v)}
                    />
                    <Input
                        label="Температура"
                        type="number"
                        value={entry.nightly_temperature}
                        onChange={(v) => handleChange("nightly_temperature", v)}
                    />
                </div>

                <button onClick={handleAddEntry} disabled={loading} className={styles.saveBtn}>
                    {loading ? "..." : <Save size={18} />}
                    Зберегти в базу
                </button>
            </Card>
        </div>
    );
};

export default Entry;
