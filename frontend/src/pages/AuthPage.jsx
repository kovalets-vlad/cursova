import React, { useState } from "react";
import { Activity } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext.jsx";
import styles from "../module_styles/AuthPage.module.css";

const AuthPage = () => {
    const { login, register } = useAuth();
    const [isLogin, setIsLogin] = useState(true);
    const navigate = useNavigate();
    const [formData, setFormData] = useState({ username: "", password: "", confirmPassword: "" });
    const [error, setError] = useState("");

    const handleChange = (name, value) => {
        setFormData((prev) => ({ ...prev, [name]: value }));
        setError("");
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError("");

        // Валідація
        if (!formData.username || !formData.password) {
            setError("Заповніть всі поля");
            return;
        }

        if (!isLogin && formData.password !== formData.confirmPassword) {
            setError("Паролі не співпадають");
            return;
        }

        const res = isLogin
            ? await login(formData.username, formData.password)
            : await register(formData.username, formData.password);

        if (res.success) {
            navigate("/", { replace: true });
        } else {
            setError(res.error);
        }
    };
    const containerClass = `${styles.container} ${!isLogin ? styles["right-panel-active"] : ""}`;

    return (
        <div className={styles.bodyWrapper}>
            <div className={containerClass} id="container">
                <div className={`${styles["form-container"]} ${styles["sign-up-container"]}`}>
                    <form className={styles.form} onSubmit={(e) => handleSubmit(e, "register")}>
                        <h1 className={styles.title}>Реєстрація</h1>
                        <span className={styles.subtitle}>Використовуйте свій логін</span>
                        <input
                            type="text"
                            placeholder="Логін"
                            className={styles.input}
                            value={formData.username}
                            onChange={(e) => handleChange("username", e.target.value)}
                        />
                        <input
                            type="password"
                            placeholder="Пароль"
                            className={styles.input}
                            value={formData.password}
                            onChange={(e) => handleChange("password", e.target.value)}
                        />
                        <input
                            type="password"
                            placeholder="Повторіть пароль"
                            className={styles.input}
                            value={formData.confirmPassword}
                            onChange={(e) => handleChange("confirmPassword", e.target.value)}
                        />
                        {error && !isLogin && <p className={styles.error}>{error}</p>}
                        <button className={styles.button}>Створити</button>
                    </form>
                </div>

                {/* --- ФОРМА ВХОДУ (Зліва в референсі) --- */}
                <div className={`${styles["form-container"]} ${styles["sign-in-container"]}`}>
                    <form className={styles.form} onSubmit={(e) => handleSubmit(e, "login")}>
                        <h1 className={styles.title}>Вхід</h1>
                        <span className={styles.subtitle}>Увійдіть у свій акаунт</span>
                        <input
                            type="text"
                            placeholder="Логін"
                            className={styles.input}
                            value={formData.username}
                            onChange={(e) => handleChange("username", e.target.value)}
                        />
                        <input
                            type="password"
                            placeholder="Пароль"
                            className={styles.input}
                            value={formData.password}
                            onChange={(e) => handleChange("password", e.target.value)}
                        />
                        <label className={styles.checkboxLabel}>
                            <input type="checkbox" /> Запом'ятати мене
                        </label>
                        {error && isLogin && <p className={styles.error}>{error}</p>}
                        <button className={styles.button}>Увійти</button>
                        <a href="#" className={styles.link}>
                            Забули пароль?
                        </a>
                    </form>
                </div>

                {/* --- OVERLAY (Рухома кольорова панель) --- */}
                <div className={styles["overlay-container"]}>
                    <div className={styles.overlay}>
                        {/* Ліва частина Overlay (видно при Реєстрації) */}
                        <div className={`${styles["overlay-panel"]} ${styles["overlay-left"]}`}>
                            <Activity size={40} className="mb-4" />
                            <h1 className={styles["title-white"]}>Вже з нами?</h1>
                            <p className={styles["overlay-text"]}>
                                Увійди в систему, щоб продовжити тренування з Pulse AI.
                            </p>
                            <button className={`${styles.button} ${styles.ghost}`} onClick={() => setIsLogin(true)}>
                                Увійти
                            </button>
                        </div>

                        {/* Права частина Overlay (видно при Вході) */}
                        <div className={`${styles["overlay-panel"]} ${styles["overlay-right"]}`}>
                            <Activity size={40} className="mb-4" />
                            <h1 className={styles["title-white"]}>Привіт, друже!</h1>
                            <p className={styles["overlay-text"]}>
                                Зареєструйся та почни свою подорож до здорового тіла.
                            </p>
                            <button className={`${styles.button} ${styles.ghost}`} onClick={() => setIsLogin(false)}>
                                Реєстрація
                            </button>
                        </div>
                    </div>
                </div>
                {/* Логотип по центру (декоративний) */}
                <div className={styles["center-logo"]}>
                    <Activity size={30} color="#dc2626" />
                </div>
            </div>
        </div>
    );
};

export default AuthPage;
