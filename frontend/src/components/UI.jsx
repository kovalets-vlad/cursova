import React from "react";

export const Card = ({ children, className = "" }) => (
    <div
        className={`relative overflow-hidden bg-slate-900/60 backdrop-blur-xl border border-slate-700/50 shadow-2xl rounded-2xl p-8 ${className}`}
    >
        <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-blue-500/20 to-transparent" />
        {children}
    </div>
);

export const Input = ({ label, type = "text", value, onChange, placeholder }) => (
    <div className="mb-5 group">
        <label className="block text-slate-400 text-xs font-bold uppercase tracking-wider mb-2 ml-1 group-focus-within:text-blue-400 transition-colors">
            {label}
        </label>
        <div className="relative">
            <input
                type={type}
                value={value}
                placeholder={placeholder}
                onChange={(e) => {
                    const val = e.target.value;
                    onChange(type === "number" ? (val === "" ? "" : parseFloat(val)) : val);
                }}
                className="w-full bg-slate-950/50 border border-slate-700 text-slate-200 rounded-xl px-4 py-3 
                focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500 focus:bg-slate-900/80
                placeholder-slate-600 outline-none transition-all duration-300 shadow-inner"
            />
        </div>
    </div>
);
