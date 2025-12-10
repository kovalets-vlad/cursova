import React, { useState, useEffect } from "react";
import { User, Lock, LogOut, Ruler, Weight, Save, Edit2, X, Calendar } from "lucide-react";
import { useAuth } from "../context/AuthContext.jsx";
import api from "../api/axiosConfig.js";
import { Card } from "../components/UI.jsx";
import styles from "../module_styles/Profile.module.css";

const Profile = () => {
    const { logout, user } = useAuth();

    const [isEditing, setIsEditing] = useState(false);
    const [loading, setLoading] = useState(false);

    const [formData, setFormData] = useState({
        weight: user?.weight || "",
        height: user?.height || "",
        age: user?.age || "",
    });

    useEffect(() => {
        if (user) {
            setFormData({
                weight: user.weight || "",
                height: user.height || "",
                age: user.age || "",
            });
        }
    }, [user]);

    const handleChange = (name, value) => {
        setFormData((prev) => ({ ...prev, [name]: value }));
    };

    const handleSave = async () => {
        setLoading(true);
        try {
            const payload = {
                weight: formData.weight ? Number(formData.weight) : null,
                height: formData.height ? Number(formData.height) : null,
                age: formData.age ? Number(formData.age) : null,
            };

            await api.patch("api/user/update", payload);

            if (user) {
                user.weight = payload.weight;
                user.height = payload.height;
                user.age = payload.age;
            }

            setIsEditing(false);
            alert("Профіль оновлено!");
        } catch (error) {
            console.error("Помилка:", error);
            alert("Помилка збереження.");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className={styles.container}>
            <Card className={styles.cardWrapper}>
                <div className={styles.decorativeBlob}></div>

                {/* --- HEADER --- */}
                <div className={styles.header}>
                    <h3 className={styles.title}>
                        <User className="text-blue-500" size={24} />
                        Мій Профіль
                    </h3>

                    {!isEditing ? (
                        <button onClick={() => setIsEditing(true)} className={`${styles.actionBtn} ${styles.editBtn}`}>
                            <Edit2 size={16} /> Редагувати
                        </button>
                    ) : (
                        <button
                            onClick={() => setIsEditing(false)}
                            className={`${styles.actionBtn} ${styles.cancelBtn}`}
                        >
                            <X size={16} /> Скасувати
                        </button>
                    )}
                </div>

                <div className={styles.contentSpace}>
                    <div className={styles.infoGrid}>
                        <div className={styles.infoBox}>
                            <div className={styles.infoLabel}>
                                <User size={16} className="text-blue-500" />
                                <span className={styles.labelText}>Логін</span>
                            </div>
                            <div className={styles.infoValue}>{user?.username}</div>
                        </div>
                    </div>

                    {/* --- Фізичні параметри (Editable) --- */}
                    <div className={styles.sectionDivider}>
                        <h4 className={styles.sectionTitle}>Фізичні параметри</h4>

                        {/* Змінив сітку на 3 колонки для десктопу, якщо місця вистачає, або залишити авто */}
                        <div className={styles.paramsGrid}>
                            {/* ВІК (Нове поле) */}
                            <div
                                className={`${styles.paramBox} ${
                                    isEditing ? styles.paramBoxEdit : styles.paramBoxView
                                }`}
                            >
                                <div className={styles.paramHeader}>
                                    <div className={styles.iconWrapper}>
                                        <Calendar size={18} className="text-purple-400" />
                                    </div>
                                    <label className="text-sm text-slate-400">Вік (років)</label>
                                </div>
                                {isEditing ? (
                                    <input
                                        type="number"
                                        value={formData.age}
                                        onChange={(e) => handleChange("age", e.target.value)}
                                        className={styles.input}
                                        placeholder="0"
                                    />
                                ) : (
                                    <div className={styles.valueDisplay}>
                                        {user?.age || <span className={styles.emptyValue}>Не вказано</span>}
                                    </div>
                                )}
                            </div>

                            {/* ВАГА */}
                            <div
                                className={`${styles.paramBox} ${
                                    isEditing ? styles.paramBoxEdit : styles.paramBoxView
                                }`}
                            >
                                <div className={styles.paramHeader}>
                                    <div className={styles.iconWrapper}>
                                        <Weight size={18} className="text-orange-400" />
                                    </div>
                                    <label className="text-sm text-slate-400">Вага (кг)</label>
                                </div>
                                {isEditing ? (
                                    <input
                                        type="number"
                                        value={formData.weight}
                                        onChange={(e) => handleChange("weight", e.target.value)}
                                        className={styles.input}
                                        placeholder="0.0"
                                    />
                                ) : (
                                    <div className={styles.valueDisplay}>
                                        {user?.weight || <span className={styles.emptyValue}>Не вказано</span>}
                                    </div>
                                )}
                            </div>

                            {/* ЗРІСТ */}
                            <div
                                className={`${styles.paramBox} ${
                                    isEditing ? styles.paramBoxEdit : styles.paramBoxView
                                }`}
                            >
                                <div className={styles.paramHeader}>
                                    <div className={styles.iconWrapper}>
                                        <Ruler size={18} className="text-teal-400" />
                                    </div>
                                    <label className="text-sm text-slate-400">Зріст (см)</label>
                                </div>
                                {isEditing ? (
                                    <input
                                        type="number"
                                        value={formData.height}
                                        onChange={(e) => handleChange("height", e.target.value)}
                                        className={styles.input}
                                        placeholder="0"
                                    />
                                ) : (
                                    <div className={styles.valueDisplay}>
                                        {user?.height || <span className={styles.emptyValue}>Не вказано</span>}
                                    </div>
                                )}
                            </div>
                        </div>

                        {/* Кнопка збереження */}
                        {isEditing && (
                            <div className={styles.saveBtnWrapper}>
                                <button onClick={handleSave} disabled={loading} className={styles.saveBtn}>
                                    {loading ? "..." : <Save size={18} />}
                                    Зберегти зміни
                                </button>
                            </div>
                        )}
                    </div>

                    {/* --- Footer (Logout) --- */}
                    <div className={styles.footer}>
                        <div className={styles.passwordBox}>
                            <div className="flex items-center gap-3">
                                <Lock size={20} className="text-slate-500" />
                                <span className="text-slate-400">Пароль</span>
                            </div>
                            <span className={styles.passwordText}>••••••••••••</span>
                        </div>

                        <button onClick={logout} className={styles.logoutBtn}>
                            <LogOut size={20} />
                            Вийти з акаунту
                        </button>
                    </div>
                </div>
            </Card>
        </div>
    );
};

export default Profile;
