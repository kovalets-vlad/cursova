import React from "react";

export const Card = ({ children, className = "" }) => (
    <div className={`bg-slate-800/50 border border-slate-700 rounded-2xl p-6 backdrop-blur-sm ${className}`}>
        {children}
    </div>
);

export const Input = ({ label, type = "text", value, onChange }) => (
    <div className="mb-4">
        <label className="block text-slate-400 text-sm mb-1">{label}</label>
        <input
            type={type}
            value={value}
            onChange={(e) => onChange(e.target.value)}
            // Важлива зміна: 'type="number"' повертає рядок, тому перетворюємо в number для безпеки
            {...(type === "number" ? { onChange: (e) => onChange(parseFloat(e.target.value) || 0) } : {})}
            className="w-full bg-slate-900 border border-slate-700 rounded-lg p-3 text-white focus:ring-2 focus:ring-blue-500 outline-none transition-all"
        />
    </div>
);
