import React, { useState } from "react";
import { Outlet, NavLink, useLocation } from "react-router-dom";
import { Activity, TrendingUp, Brain, PlusCircle, User, LogOut, Menu, X } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext.jsx";
import styles from "../module_styles/DashboardLayout.module.css";

const DashboardLayout = () => {
    const { logout, user } = useAuth();
    const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
    const navigate = useNavigate();
    const location = useLocation();

    const navItems = [
        { path: "/", icon: TrendingUp, label: "Огляд" },
        { path: "/forecast", icon: Brain, label: "AI Прогноз" },
        { path: "/entry", icon: PlusCircle, label: "Додати запис" },
        { path: "/profile", icon: User, label: "Профіль" },
    ];

    // Знаходимо заголовок для поточної сторінки
    const currentTitle = navItems.find((item) => item.path === location.pathname)?.label || "Панель керування";

    const toggleMenu = () => setIsMobileMenuOpen(!isMobileMenuOpen);
    const closeMenu = () => setIsMobileMenuOpen(false);

    return (
        <div className={styles.layoutContainer}>
            {/* Мобільна навігація (Overlay) */}
            <div
                className={`${styles.mobileOverlay} ${isMobileMenuOpen ? styles.mobileOverlayActive : ""}`}
                onClick={closeMenu}
            ></div>

            {/* --- SIDEBAR --- */}
            <aside className={`${styles.sidebar} ${isMobileMenuOpen ? styles.sidebarActive : ""}`}>
                <div className={styles.logoContainer}>
                    <div className={styles.logoIconWrapper}>
                        <Activity className={styles.logoIcon} />
                    </div>
                    <span className={styles.logoText}>Pulse AI</span>
                    <button className={styles.mobileCloseBtn} onClick={closeMenu}>
                        <X size={24} />
                    </button>
                </div>

                <nav className={styles.nav}>
                    {navItems.map((item) => (
                        <NavLink
                            key={item.path}
                            to={item.path}
                            onClick={closeMenu}
                            className={({ isActive }) => `${styles.navItem} ${isActive ? styles.navItemActive : ""}`}
                        >
                            <item.icon className={styles.navIcon} size={20} />
                            <span className={styles.navLabel}>{item.label}</span>
                            {/* Декоративна лінія активного стану */}
                            <div className={styles.activeIndicator}></div>
                        </NavLink>
                    ))}
                </nav>

                <div className={styles.footer}>
                    <button onClick={logout} className={styles.logoutBtn}>
                        <LogOut size={20} />
                        <span>Вихід</span>
                    </button>
                </div>
            </aside>

            {/* --- MAIN CONTENT --- */}
            <div className={styles.mainWrapper}>
                {/* --- HEADER --- */}
                <header className={styles.header}>
                    <div className={styles.headerLeft}>
                        <button className={styles.burgerBtn} onClick={toggleMenu}>
                            <Menu size={24} />
                        </button>
                        <h2 className={styles.pageTitle}>{currentTitle}</h2>
                    </div>

                    <div className={styles.userProfile} onClick={() => navigate("/profile")}>
                        <div className={styles.avatar}>{user?.username?.[0].toUpperCase()}</div>
                        <span className={styles.username}>{user?.username}</span>
                    </div>
                </header>

                {/* --- CONTENT AREA --- */}
                <main className={styles.contentArea}>
                    <Outlet />
                </main>
            </div>
        </div>
    );
};

export default DashboardLayout;
