import React from "react";
import styles from "../module_styles/Input.module.css";

export const Card = ({ children, className = "" }) => (
    <div
        className={`relative overflow-hidden bg-slate-900/60 backdrop-blur-xl border border-slate-700/50 shadow-2xl rounded-2xl p-8 ${className}`}
    >
        <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-blue-500/20 to-transparent" />
        {children}
    </div>
);

// components/UI.jsx
export const Input = ({ label, type = "text", value, onChange, disabled, placeholder, index, isLast }) => (
    <div className={styles.inputGroup}>
        {/* Нумерація */}
        {index && (
            <div className={styles.stepWrapper}>
                <div className={styles.stepCircle}>{index}</div>
                {/* Лінія між кроками (ховаємо, якщо це останній інпут) */}
                {!isLast && <div className={styles.stepLine} />}
            </div>
        )}

        <div className={styles.fieldContainer}>
            {label && <label className={styles.label}>{label}</label>}
            <input
                type={type}
                value={value}
                disabled={disabled}
                placeholder={placeholder || label}
                onChange={(e) => onChange(e.target.value)}
                className={styles.inputField}
            />
        </div>
    </div>
);
