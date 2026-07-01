import pygame
import sys
import math
import random
import numpy as np
import csv
import datetime
import requests
import threading
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

# Импорты модулей проекта
from environment.beacon import Beacon
from robot.robot import Robot
from navigation.ekf import EKF


# --- ЗАГРУЗКА КОНФИГУРАЦИИ ---
def load_config():
    default_conf = {
        "window": {"width": 1920, "height": 1080, "fps": 60},
        "telegram": {"enabled": False, "token": "", "chat_id": ""},
        "simulation": {"beacon_range": 800, "obstacles_count": 7, "tractor_speed": 160, "tractor_width_px": 60,
                       "tractor_offset_px": 30},
        "lidar": {"range": 350, "fov": 360, "stop_distance": 40}
    }
    if not os.path.exists("settings.json"):
        return default_conf
    try:
        with open("settings.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default_conf


CONFIG = load_config()

# --- ПАРАМЕТРЫ И ГЕОМЕТРИЯ ---
WINDOW_W = CONFIG["window"]["width"]
WINDOW_H = CONFIG["window"]["height"]
FPS = CONFIG["window"]["fps"]
TAB_HEIGHT = 40
SIDEBAR_W = 450
FIELD_AREA_W = WINDOW_W - SIDEBAR_W
FIELD_AREA_H = WINDOW_H - TAB_HEIGHT

BEACON_COMMS_RANGE = CONFIG["simulation"]["beacon_range"]

# === НАСТРОЙКИ TELEGRAM — читаем из settings.json ===
# ИСПРАВЛЕНИЕ: раньше токены были захардкожены как "ВСТАВЬ_СЮДА_ТОКЕН",
# теперь они корректно читаются из settings.json
TG_BOT_TOKEN = CONFIG["telegram"].get("token", "")
TG_CHAT_ID   = CONFIG["telegram"].get("chat_id", "")
TG_ENABLED   = CONFIG["telegram"].get("enabled", False)
# =====================================================

IMPLEMENT_WIDTH = CONFIG["simulation"]["tractor_width_px"]
IMPLEMENT_OFFSET = CONFIG["simulation"]["tractor_offset_px"]
GRID_CELL_SIZE = 5

LIDAR_RANGE = CONFIG["lidar"]["range"]
LIDAR_FOV = 360
LIDAR_RAYS = 16
STOP_DISTANCE = 45
AVOID_DISTANCE = 250

# --- ЦВЕТОВАЯ ПАЛИТРА ---
C_BG         = (18, 18, 22)
C_PANEL      = (25, 25, 30)
C_GRID       = (40, 45, 50)
C_TEXT_MAIN  = (240, 240, 245)
C_TEXT_DIM   = (140, 140, 150)
C_ACCENT     = (0, 150, 255)
C_WARN       = (255, 170, 0)
C_ERR        = (255, 60, 80)
C_GOOD       = (40, 200, 100)
C_COVERAGE   = (0, 180, 120)
# ИСПРАВЛЕНИЕ: C_GOLD отсутствовал в main.py, но использовался → NameError
C_GOLD       = (255, 215, 0)

# --- МАШИННОЕ ОБУЧЕНИЕ (ПАМЯТЬ) ---
MEMORY_FILE = "dist/ai_memory.json"


def load_ai_memory():
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return {"obstacles": data, "learned_path": []}
                return data
        except Exception:
            pass
    return {"obstacles": [], "learned_path": []}


def save_ai_memory(memory_dict):
    with open(MEMORY_FILE, 'w') as f:
        json.dump(memory_dict, f)


# --- АНАЛИТИКА ---
def open_file_and_analyze():
    root = tk.Tk()
    root.withdraw()
    curr_dir = os.getcwd()
    file_path = filedialog.askopenfilename(
        initialdir=curr_dir, title="Выберите лог",
        filetypes=[("CSV Files", "*.csv")])
    root.destroy()
    if not file_path:
        return
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Ошибка чтения: {e}")
        return

    df['Time_s'] = (df['Time_ms'] - df['Time_ms'].iloc[0]) / 1000.0
    df_moving = df[df['Speed_px_s'] > 5] if 'Speed_px_s' in df.columns else df

    x_col    = 'X'
    y_col    = 'Y'
    err_col  = 'Error_px'
    dr_col   = 'DR_Error_px' if 'DR_Error_px' in df.columns else None
    err_unit = 'px'

    ev_avoid   = df[df['Event'] == 'AVOIDANCE']   if 'Event' in df.columns else pd.DataFrame()
    ev_reverse = df[df['Event'] == 'REVERSE']     if 'Event' in df.columns else pd.DataFrame()
    ev_bfail   = df[df['Event'] == 'BEACON_FAIL'] if 'Event' in df.columns else pd.DataFrame()

    plt.style.use('dark_background')
    fig = plt.figure(figsize=(18, 11))
    fig.patch.set_facecolor('#1a1a2e')
    fig.suptitle(
        f'AgroNav — Анализ сессии: {os.path.basename(file_path)}',
        fontsize=15, color='#00d4ff', fontweight='bold', y=0.98)

    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35)
    ax_path  = fig.add_subplot(gs[0:2, 0])
    ax_err   = fig.add_subplot(gs[0, 1:])
    ax_gdop  = fig.add_subplot(gs[1, 1])
    ax_cov   = fig.add_subplot(gs[1, 2])
    ax_box   = fig.add_subplot(gs[2, 0])
    ax_hist  = fig.add_subplot(gs[2, 1])
    ax_stats = fig.add_subplot(gs[2, 2])

    PANEL  = '#16213e'
    ACCENT = '#00d4ff'
    GREEN  = '#00ff88'
    RED    = '#ff4444'
    ORANGE = '#ffaa00'
    YELLOW = '#ffff44'

    for ax in [ax_path, ax_err, ax_gdop, ax_cov, ax_box, ax_hist, ax_stats]:
        ax.set_facecolor(PANEL)
        for sp in ax.spines.values():
            sp.set_color('#333355')

    # 1. Траектория
    ax_path.set_title('Траектория трактора', color=ACCENT, pad=8)
    ax_path.plot(df[x_col], df[y_col], color='#3388ff', lw=1.5, label='Путь', zorder=2)
    ax_path.plot(df[x_col].iloc[0], df[y_col].iloc[0], 'o', color=GREEN, ms=8, label='Старт', zorder=5)
    ax_path.plot(df[x_col].iloc[-1], df[y_col].iloc[-1], 's', color=RED, ms=8, label='Финиш', zorder=5)
    if not ev_avoid.empty and x_col in ev_avoid.columns:
        ax_path.scatter(ev_avoid[x_col], ev_avoid[y_col],
                        c=ORANGE, s=55, marker='^', label='Объезд', zorder=6, alpha=0.85)
    if not ev_reverse.empty and x_col in ev_reverse.columns:
        ax_path.scatter(ev_reverse[x_col], ev_reverse[y_col],
                        c=RED, s=55, marker='v', label='Реверс', zorder=6, alpha=0.85)
    if not ev_bfail.empty and x_col in ev_bfail.columns:
        ax_path.scatter(ev_bfail[x_col], ev_bfail[y_col],
                        c=YELLOW, s=55, marker='x', label='Отказ маяка', zorder=6, alpha=0.85)
    ax_path.invert_yaxis()
    ax_path.legend(fontsize=7, loc='lower right', facecolor='#0d0d1a', labelcolor='white')
    ax_path.set_xlabel('X (px)', color='gray', fontsize=8)
    ax_path.set_ylabel('Y (px)', color='gray', fontsize=8)
    ax_path.tick_params(colors='gray', labelsize=7)

    # 2. Ошибка EKF по времени
    ax_err.set_title('Ошибка позиционирования: EKF vs Dead Reckoning', color=ACCENT, pad=8)
    ax_err.plot(df['Time_s'], df[err_col], color=RED, lw=1.5, label='Ошибка EKF', zorder=3)
    ax_err.fill_between(df['Time_s'], df[err_col], alpha=0.12, color=RED)
    if dr_col and dr_col in df.columns:
        ax_err.plot(df['Time_s'], df[dr_col],
                    color=ORANGE, lw=1.2, ls='--', label='Ошибка DR (без EKF)', zorder=2, alpha=0.85)
        ax_err.fill_between(df['Time_s'], df[dr_col], alpha=0.07, color=ORANGE)
    for _, row in ev_avoid.iterrows():
        ax_err.axvline(row['Time_s'], color=ORANGE, lw=0.8, alpha=0.6)
    for _, row in ev_reverse.iterrows():
        ax_err.axvline(row['Time_s'], color=RED, lw=0.8, alpha=0.5)
    for _, row in ev_bfail.iterrows():
        ax_err.axvline(row['Time_s'], color=YELLOW, lw=0.8, alpha=0.6)
    ax_err.axhline(10, color=GREEN,  lw=0.7, ls='--', alpha=0.7, label='Отлично (<10px)')
    ax_err.axhline(25, color=ORANGE, lw=0.7, ls='--', alpha=0.5, label='Норма (<25px)')
    ax_err.set_xlabel('Время (с)', color='gray', fontsize=8)
    ax_err.set_ylabel('Ошибка (px)', color='gray', fontsize=8)
    ax_err.legend(fontsize=7, facecolor='#0d0d1a', labelcolor='white')
    ax_err.tick_params(colors='gray', labelsize=7)

    # 3. GDOP по времени
    ax_gdop.set_title('GDOP (геометрия маяков)', color=ACCENT, pad=8)
    if 'GDOP' in df.columns:
        gdop_clip = df['GDOP'].clip(upper=15)
        ax_gdop.plot(df['Time_s'], gdop_clip, color=YELLOW, lw=1.2)
        ax_gdop.fill_between(df['Time_s'], gdop_clip, alpha=0.15, color=YELLOW)
        ax_gdop.axhline(2, color=GREEN,  lw=0.7, ls='--', alpha=0.7, label='Отлично (<2)')
        ax_gdop.axhline(4, color=ORANGE, lw=0.7, ls='--', alpha=0.7, label='Норма (<4)')
        ax_gdop.axhline(8, color=RED,    lw=0.7, ls='--', alpha=0.7, label='Плохо (>8)')
        ax_gdop.legend(fontsize=7, facecolor='#0d0d1a', labelcolor='white')
    ax_gdop.set_xlabel('Время (с)', color='gray', fontsize=8)
    ax_gdop.set_ylabel('GDOP', color='gray', fontsize=8)
    ax_gdop.tick_params(colors='gray', labelsize=7)

    # 4. Ковариация EKF
    ax_cov.set_title('Неопределённость позиции EKF', color=ACCENT, pad=8)
    if 'EKF_Cov' in df.columns:
        ax_cov.plot(df['Time_s'], df['EKF_Cov'], color='#cc44ff', lw=1.2)
        ax_cov.fill_between(df['Time_s'], df['EKF_Cov'], alpha=0.15, color='#cc44ff')
    ax_cov.set_xlabel('Время (с)', color='gray', fontsize=8)
    ax_cov.set_ylabel('trace(P)', color='gray', fontsize=8)
    ax_cov.tick_params(colors='gray', labelsize=7)

    # 5. Ошибка vs число маяков
    ax_box.set_title('Ошибка EKF vs число маяков', color=ACCENT, pad=8)
    if 'Visible_Beacons' in df.columns:
        uniq = sorted(df['Visible_Beacons'].unique())
        groups = [df[df['Visible_Beacons'] == n]['Error_px'].dropna().values for n in uniq]
        labels = [f"{int(n)} маяк." for n in uniq]
        bp = ax_box.boxplot(groups, labels=labels, patch_artist=True,
                            medianprops=dict(color=ACCENT, lw=2))
        box_colors = [GREEN, YELLOW, ORANGE, RED, '#cc44ff']
        for patch, c in zip(bp['boxes'], box_colors):
            patch.set_facecolor(c); patch.set_alpha(0.6)
    ax_box.set_ylabel('Ошибка (px)', color='gray', fontsize=8)
    ax_box.tick_params(colors='gray', labelsize=7)

    # 6. Гистограмма ошибки
    ax_hist.set_title('Распределение ошибки EKF', color=ACCENT, pad=8)
    ax_hist.hist(df_moving['Error_px'], bins=25, color=ORANGE, edgecolor='#333355', alpha=0.85)
    mean_err = df_moving['Error_px'].mean()
    ax_hist.axvline(mean_err, color=RED, lw=1.5, ls='--', label=f'Среднее: {mean_err:.1f}px')
    ax_hist.legend(fontsize=7, facecolor='#0d0d1a', labelcolor='white')
    ax_hist.set_xlabel('Ошибка (px)', color='gray', fontsize=8)
    ax_hist.set_ylabel('Кадры', color='gray', fontsize=8)
    ax_hist.tick_params(colors='gray', labelsize=7)

    # 7. Итоговая таблица
    ax_stats.axis('off')
    ax_stats.set_title('Итоги сессии', color=ACCENT, pad=8)
    duration  = df['Time_s'].iloc[-1]
    err_mean  = df[err_col].mean()
    err_max   = df[err_col].max()
    err_p95   = df[err_col].quantile(0.95)
    dr_mean   = df[dr_col].mean()  if dr_col and dr_col in df.columns else float('nan')
    dr_max    = df[dr_col].max()   if dr_col and dr_col in df.columns else float('nan')
    gdop_mean = df['GDOP'].mean()  if 'GDOP' in df.columns else float('nan')
    pct_good  = (df[err_col] < 10).mean() * 100
    gain_mean = (dr_mean - err_mean) if not math.isnan(dr_mean) else float('nan')
    rows = [
        ["Длительность",            f"{duration:.1f} с"],
        ["── EKF ──────────────",   ""],
        ["Ошибка EKF средняя",      f"{err_mean:.2f} px"],
        ["Ошибка EKF макс.",        f"{err_max:.2f} px"],
        ["Ошибка EKF P95",          f"{err_p95:.2f} px"],
        ["Кадров ошибка <10px",     f"{pct_good:.1f}%"],
        ["── DR (без EKF) ──────",  ""],
        ["Ошибка DR средняя",       f"{dr_mean:.2f} px" if not math.isnan(dr_mean) else "—"],
        ["Ошибка DR макс.",         f"{dr_max:.2f} px"  if not math.isnan(dr_max)  else "—"],
        ["── Сравнение ──────────", ""],
        ["EKF точнее DR на",        f"{gain_mean:.1f} px" if not math.isnan(gain_mean) else "—"],
        ["GDOP средний",            f"{gdop_mean:.2f}" if not math.isnan(gdop_mean) else "—"],
        ["Манёвров объезда",        str(len(ev_avoid))],
        ["Экстренных реверсов",     str(len(ev_reverse))],
        ["Отказов маяков",          str(len(ev_bfail))],
    ]
    table = ax_stats.table(
        cellText=rows, colLabels=["Показатель", "Значение"],
        cellLoc='left', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    for (r, c), cell in table.get_celld().items():
        cell.set_facecolor('#0d0d1a' if r > 0 else '#1a1a3e')
        cell.set_edgecolor('#333355')
        cell.set_text_props(color='white' if r > 0 else ACCENT)

    plt.show()


def draw_analysis_screen(screen, font_title, font_main, font_sm, err_history, dr_err_history,
                         gdop_history, cov_history, log, rec, avoid_cnt):
    """Вкладка с живой навигационной аналитикой и внешним CSV-анализом."""
    screen.fill((16, 16, 20))

    def mean(values):
        return sum(values) / len(values) if values else 0.0

    def metric(x, y, label, value, color=C_TEXT_MAIN):
        screen.blit(font_main.render(label, True, C_TEXT_DIM), (x, y))
        screen.blit(font_main.render(value, True, color), (x + 250, y))

    current_ekf = err_history[-1] if err_history else 0.0
    current_dr = dr_err_history[-1] if dr_err_history else 0.0
    avg_ekf = mean(err_history)
    avg_dr = mean(dr_err_history)
    avg_gain = avg_dr - avg_ekf
    valid_gdop = [v for v in gdop_history if v < 90]
    avg_gdop = mean(valid_gdop)
    current_gdop = valid_gdop[-1] if valid_gdop else 99.0
    current_cov = cov_history[-1] if cov_history else 0.0

    screen.blit(font_title.render("АНАЛИТИКА ПРОЙДЕННОГО ПУТИ", True, C_ACCENT), (40, 70))
    screen.blit(
        font_main.render("Сравнение EKF и одометрии вынесено с основного экрана, чтобы симулятор оставался читаемым.",
                         True, C_TEXT_DIM),
        (40, 108)
    )

    main_card = pygame.Rect(40, 150, 980, 360)
    draw_card(screen, main_card)
    screen.blit(font_title.render("Ошибка позиционирования: EKF vs DR", True, C_ACCENT),
                (main_card.x + 15, main_card.y + 14))
    pygame.draw.circle(screen, (0, 210, 255), (main_card.x + 18, main_card.y + 36), 4)
    screen.blit(font_sm.render("EKF", True, C_TEXT_DIM), (main_card.x + 30, main_card.y + 29))
    pygame.draw.circle(screen, C_WARN, (main_card.x + 80, main_card.y + 36), 4)
    screen.blit(font_sm.render("Одометрия", True, C_TEXT_DIM), (main_card.x + 92, main_card.y + 29))
    draw_live_chart(
        screen,
        pygame.Rect(main_card.x + 15, main_card.y + 58, main_card.w - 30, main_card.h - 78),
        [
            (err_history, (0, 210, 255), True),
            (dr_err_history, C_WARN, False),
        ],
        thresholds=[(25, C_WARN, "< 25 px"), (10, C_GOOD, "< 10 px")]
    )

    gdop_card = pygame.Rect(1050, 150, 420, 170)
    draw_card(screen, gdop_card)
    screen.blit(font_main.render("GDOP во времени", True, C_ACCENT), (gdop_card.x + 15, gdop_card.y + 14))
    pygame.draw.circle(screen, C_GOLD, (gdop_card.x + 18, gdop_card.y + 35), 4)
    screen.blit(font_sm.render("GDOP", True, C_TEXT_DIM), (gdop_card.x + 30, gdop_card.y + 28))
    draw_live_chart(
        screen,
        pygame.Rect(gdop_card.x + 15, gdop_card.y + 48, gdop_card.w - 30, gdop_card.h - 65),
        [(gdop_history, C_GOLD, False)],
        thresholds=[(2, C_ERR, "2")]
    )

    cov_card = pygame.Rect(1050, 340, 420, 170)
    draw_card(screen, cov_card)
    screen.blit(font_main.render("Неопределённость EKF", True, C_ACCENT), (cov_card.x + 15, cov_card.y + 14))
    pygame.draw.circle(screen, (205, 80, 255), (cov_card.x + 18, cov_card.y + 35), 4)
    screen.blit(font_sm.render("trace(P)", True, C_TEXT_DIM), (cov_card.x + 30, cov_card.y + 28))
    draw_live_chart(
        screen,
        pygame.Rect(cov_card.x + 15, cov_card.y + 48, cov_card.w - 30, cov_card.h - 65),
        [(cov_history, (205, 80, 255), False)]
    )

    session_card = pygame.Rect(40, 540, 980, 260)
    draw_card(screen, session_card)
    screen.blit(font_title.render("ТЕКУЩАЯ СЕССИЯ", True, C_ACCENT), (session_card.x + 15, session_card.y + 18))
    left_x = session_card.x + 20
    right_x = session_card.x + 500
    row_y = session_card.y + 68
    metric(left_x, row_y, "Текущая ошибка EKF", f"{current_ekf:.2f} px", C_GOOD if current_ekf < 10 else C_WARN)
    metric(left_x, row_y + 38, "Текущая ошибка одометрии", f"{current_dr:.2f} px", C_WARN)
    metric(left_x, row_y + 76, "Средняя ошибка EKF", f"{avg_ekf:.2f} px")
    metric(left_x, row_y + 114, "Средняя ошибка одометрии", f"{avg_dr:.2f} px")
    metric(left_x, row_y + 152, "Средний выигрыш EKF", f"{avg_gain:+.2f} px",
           C_GOOD if avg_gain >= 0 else C_ERR)

    metric(right_x, row_y, "Текущий GDOP", f"{current_gdop:.2f}", C_GOOD if current_gdop < 2 else C_WARN)
    metric(right_x, row_y + 38, "Средний GDOP", f"{avg_gdop:.2f}" if valid_gdop else "—")
    metric(right_x, row_y + 76, "Текущая trace(P)", f"{current_cov:.3f}")
    metric(right_x, row_y + 114, "Манёвров объезда", str(avoid_cnt))
    metric(right_x, row_y + 152, "Точек в CSV-буфере", str(len(log)))
    screen.blit(font_sm.render("",
                               True, C_TEXT_DIM),
                (session_card.x + 20, session_card.bottom - 30))

    external_card = pygame.Rect(1050, 540, 420, 260)
    draw_card(screen, external_card)
    screen.blit(font_title.render("ВНЕШНИЙ АНАЛИЗ", True, C_ACCENT), (external_card.x + 15, external_card.y + 18))
    screen.blit(font_main.render("Откройте сохранённый CSV-файл, чтобы построить полный набор графиков",
                                 True, C_TEXT_DIM), (external_card.x + 15, external_card.y + 68))
    screen.blit(font_main.render("Этот режим удобен для скриншотов",
                                 True, C_TEXT_DIM), (external_card.x + 15, external_card.y + 104))
    log_state = "Запись лога включена" if rec else "Запись лога выключена"
    screen.blit(font_main.render(log_state, True, C_GOOD if rec else C_TEXT_DIM),
                (external_card.x + 15, external_card.bottom - 70))
    screen.blit(font_main.render("Симулятор: [S] начать/остановить запись", True, C_TEXT_DIM),
                (external_card.x + 15, external_card.bottom - 35))

    btn_rect = get_analysis_button_rect()
    pygame.draw.rect(screen, C_ACCENT, btn_rect, border_radius=8)
    btn_text = font_main.render("ОТКРЫТЬ CSV-ЛОГ", True, C_BG)
    screen.blit(btn_text, btn_text.get_rect(center=btn_rect.center))


# --- TELEGRAM ---
def trigger_alert(message):
    if not TG_ENABLED or len(TG_BOT_TOKEN) < 10:
        return

    def send():
        try:
            requests.post(
                f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage",
                data={"chat_id": TG_CHAT_ID, "text": f"🤖 Agro AI:\n{message}"},
                timeout=5
            )
        except Exception:
            pass

    threading.Thread(target=send, daemon=True).start()


# --- МАТЕМАТИКА ---
def segment_circle_intersection(A, B, C, R):
    Ax, Ay = A
    Bx, By = B
    Cx, Cy = C
    Dx, Dy = Bx - Ax, By - Ay
    Fx, Fy = Cx - Ax, Cy - Ay
    AB_len_sq = Dx * Dx + Dy * Dy
    if AB_len_sq == 0:
        return math.hypot(Fx, Fy) <= R
    t = max(0, min(1, (Fx * Dx + Fy * Dy) / AB_len_sq))
    return math.hypot(Cx - (Ax + t * Dx), Cy - (Ay + t * Dy)) <= R


def clamp_pt(p):
    x = max(40, min(FIELD_AREA_W - 40, p[0]))
    y = max(TAB_HEIGHT + 40, min(FIELD_AREA_H + TAB_HEIGHT - 40, p[1]))
    return (x, y)


def smooth_path(path, iterations=2):
    if len(path) < 3:
        return path
    for _ in range(iterations):
        new_path = [path[0]]
        for i in range(len(path) - 1):
            p0 = path[i]
            p1 = path[i + 1]
            new_path.append((p0[0] * 0.75 + p1[0] * 0.25, p0[1] * 0.75 + p1[1] * 0.25))
            new_path.append((p0[0] * 0.25 + p1[0] * 0.75, p0[1] * 0.25 + p1[1] * 0.75))
        new_path.append(path[-1])
        path = new_path
    return path


# --- КЛАССЫ СЕНСОРОВ ---
class SectorLidar:
    def __init__(self, radius=200):
        self.radius = radius
        self.sectors = LIDAR_RAYS
        self.sector_angles = 360 / self.sectors
        self.distances = [radius] * self.sectors
        self.visible = True

    def scan(self, rx, ry, rt, obstacles):
        self.distances = [self.radius] * self.sectors
        for angle_deg in range(0, 360, 5):
            angle_rad = rt + math.radians(angle_deg)
            sector_idx = int(((angle_deg + self.sector_angles / 2) % 360) // self.sector_angles)
            ray_dir = (math.cos(angle_rad), math.sin(angle_rad))
            closest = self.radius
            for obs in obstacles:
                oc = (obs.x - rx, obs.y - ry)
                tc = oc[0] * ray_dir[0] + oc[1] * ray_dir[1]
                if tc > 0:
                    d2 = (oc[0] ** 2 + oc[1] ** 2) - tc ** 2
                    if d2 <= obs.radius ** 2:
                        dist = tc - math.sqrt(obs.radius ** 2 - d2)
                        if 0 < dist < closest:
                            closest = dist
            if closest < self.distances[sector_idx]:
                self.distances[sector_idx] = closest

    def draw(self, screen, rx, ry, rt):
        if not self.visible:
            return
        surf = pygame.Surface((self.radius * 2, self.radius * 2), pygame.SRCALPHA)
        center = (self.radius, self.radius)
        for i in range(self.sectors):
            dist = self.distances[i]
            if dist == self.radius:
                continue
            if dist < 45:
                color = (255, 50, 50, 100)
            elif dist < 80:
                color = (255, 150, 0, 80)
            elif dist < 130:
                color = (255, 255, 0, 50)
            else:
                color = (0, 255, 100, 30)
            start_a = math.radians(i * self.sector_angles - self.sector_angles / 2)
            end_a = math.radians(i * self.sector_angles + self.sector_angles / 2)
            points = [center]
            for j in range(6):
                a = start_a + (end_a - start_a) * (j / 5)
                points.append((center[0] + math.cos(a) * dist, center[1] + math.sin(a) * dist))
            pygame.draw.polygon(surf, color, points)
            pygame.draw.polygon(surf, (color[0], color[1], color[2], 200), points, 1)
        rot_surf = pygame.transform.rotate(surf, -math.degrees(rt))
        screen.blit(rot_surf, rot_surf.get_rect(center=(rx, ry)))

    def get_front_distance(self):
        return min(self.distances[0], self.distances[1], self.distances[-1])


class Obstacle:
    def __init__(self, x, y, t):
        self.x = x
        self.y = y
        self.type = t
        self.radius = 25 if t == 'rock' else 40
        self.rect = pygame.Rect(x - self.radius, y - self.radius, self.radius * 2, self.radius * 2)
        self.known = False
        self.tg_alerted = False

    def draw(self, scr):
        c1, c2 = ((60, 60, 65), (80, 80, 85)) if self.type == 'rock' else ((20, 80, 40), (30, 100, 50))
        pygame.draw.circle(scr, c1, (self.x, self.y), self.radius)
        pygame.draw.circle(scr, c2, (self.x - 3, self.y - 3), self.radius - 4)
        if self.known:
            pygame.draw.rect(scr, C_ERR, self.rect.inflate(20, 20), 2, border_radius=4)
            pygame.draw.line(scr, C_ERR, (self.x - 10, self.y), (self.x + 10, self.y), 2)
            pygame.draw.line(scr, C_ERR, (self.x, self.y - 10), (self.x, self.y + 10), 2)


class CoverageManager:
    def __init__(self, w, h):
        self.surf = pygame.Surface((w, h), pygame.SRCALPHA)
        self.w = w
        self.h = h
        self.cells = set()

    def paint(self, x, y, a):
        ca, sa = math.cos(a), math.sin(a)
        cx, cy = x - ca * IMPLEMENT_OFFSET, y - sa * IMPLEMENT_OFFSET
        lx, ly = cx + sa * (IMPLEMENT_WIDTH / 2), cy - ca * (IMPLEMENT_WIDTH / 2)
        rx, ry = cx - sa * (IMPLEMENT_WIDTH / 2), cy + ca * (IMPLEMENT_WIDTH / 2)
        pygame.draw.circle(self.surf, C_COVERAGE + (60,), (int(lx), int(ly)), 5)
        pygame.draw.circle(self.surf, C_COVERAGE + (60,), (int(rx), int(ry)), 5)
        pygame.draw.line(self.surf, C_COVERAGE + (60,), (lx, ly), (rx, ry), 10)
        steps = int(IMPLEMENT_WIDTH / GRID_CELL_SIZE) + 1
        for i in range(steps):
            t = i / steps
            px, py = lx + (rx - lx) * t, ly + (ry - ly) * t
            if (0 <= int(px // GRID_CELL_SIZE) < self.w // GRID_CELL_SIZE and
                    0 <= int(py // GRID_CELL_SIZE) < self.h // GRID_CELL_SIZE):
                self.cells.add((int(px // GRID_CELL_SIZE), int(py // GRID_CELL_SIZE)))

    def draw(self, scr):
        scr.blit(self.surf, (0, TAB_HEIGHT))

    def get_hectares(self):
        return len(self.cells) / 8000.0


# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---
def calculate_mesh_network(beacons):
    for b in beacons:
        b.neighbors = []
    for i in range(len(beacons)):
        for j in range(i + 1, len(beacons)):
            b1, b2 = beacons[i], beacons[j]
            if math.hypot(b1.x - b2.x, b1.y - b2.y) <= BEACON_COMMS_RANGE:
                b1.neighbors.append(b2)
                b2.neighbors.append(b1)


def compute_gdop(px, py, visible_beacons):
    """
    GDOP — Geometric Dilution of Precision.
    Стандартная метрика GPS/UWB навигации. Показывает насколько хороша
    геометрия расстановки маяков в точке (px, py).
    Формула: GDOP = sqrt(trace( (H^T * H)^-1 ))
    где H — матрица единичных векторов направлений к маякам.
    Чем меньше GDOP — тем лучше геометрия:
      GDOP < 2   — отличная геометрия (маяки вокруг)
      GDOP 2–4   — хорошая
      GDOP 4–8   — удовлетворительная
      GDOP > 8   — плохая (маяки кластеризованы или мало маяков)
    """
    if len(visible_beacons) < 2:
        return 99.0  # недостаточно маяков — позиционирование невозможно
    rows = []
    for b in visible_beacons:
        dist = math.hypot(px - b.x, py - b.y)
        if dist < 1e-6:
            continue
        rows.append([(px - b.x) / dist, (py - b.y) / dist])
    if len(rows) < 2:
        return 99.0
    H = np.array(rows)
    try:
        HtH_inv = np.linalg.inv(H.T @ H)
        return float(math.sqrt(max(0, np.trace(HtH_inv))))
    except np.linalg.LinAlgError:
        return 99.0


def generate_heatmap(w, h, beacons):
    """
    GDOP-карта вместо простой карты покрытия.
    Цвет ячейки показывает качество позиционирования:
      Зелёный  — GDOP < 2   (отличная геометрия)
      Жёлтый   — GDOP 2–4   (хорошая)
      Оранжевый— GDOP 4–8   (удовлетворительная)
      Красный  — GDOP > 8   (плохая или нет маяков)
    """
    surf = pygame.Surface((w, h), pygame.SRCALPHA)
    step = 30
    active_bcs = [b for b in beacons if b.active]
    for x in range(0, w, step):
        for y in range(0, h, step):
            # Маяки, видимые из этой точки
            visible = [b for b in active_bcs if math.hypot(x - b.x, y - b.y) <= b.radius]
            gdop = compute_gdop(x, y, visible)
            if gdop < 2.0:
                col = (0, 220, 80, 45)    # отличная геометрия — зелёный
            elif gdop < 4.0:
                col = (180, 255, 0, 40)   # хорошая — жёлто-зелёный
            elif gdop < 8.0:
                col = (255, 160, 0, 40)   # удовлетворительная — оранжевый
            else:
                col = (255, 40, 40, 45)   # плохая / нет маяков — красный
            pygame.draw.rect(surf, col, (x, y, step, step))
    return surf
    eig_val, eig_vec = np.linalg.eig(P[0:2, 0:2])
    if eig_val[0] >= eig_val[1]:
        big, small = 0, 1
    else:
        big, small = 1, 0
    t = math.atan2(eig_vec[1, big], eig_vec[0, big])
    w = 2 * math.sqrt(max(0, eig_val[big])) * 40
    h = 2 * math.sqrt(max(0, eig_val[small])) * 40
    return w, h, math.degrees(t)


def create_dirt(w, h):
    s = pygame.Surface((w, h))
    s.fill((28, 28, 32))
    for _ in range(5000):
        pygame.draw.circle(s, (35, 35, 40), (random.randint(0, w), random.randint(0, h)), random.randint(1, 2))
    return s


def create_tractor(crashed):
    s = pygame.Surface((44, 60), pygame.SRCALPHA)
    col = (200, 50, 50) if crashed else (255, 140, 0)
    for y in [2, 38]:
        pygame.draw.rect(s, (10, 10, 10), (0, y, 12, 20), border_radius=2)
        pygame.draw.rect(s, (10, 10, 10), (32, y, 12, 20), border_radius=2)
    pygame.draw.rect(s, col, (10, 6, 24, 48), border_radius=4)
    pygame.draw.rect(s, (30, 40, 50), (14, 20, 16, 18), border_radius=2)
    if crashed:
        pygame.draw.line(s, (255, 255, 255), (14, 20), (30, 38), 2)
        pygame.draw.line(s, (255, 255, 255), (30, 20), (14, 38), 2)
    return s


def create_beacon(col):
    s = pygame.Surface((40, 40), pygame.SRCALPHA)
    pygame.draw.polygon(s, (60, 60, 65), [(20, 35), (12, 40), (28, 40)])
    pygame.draw.line(s, col, (12, 35), (20, 10), 2)
    pygame.draw.line(s, col, (28, 35), (20, 10), 2)
    pygame.draw.circle(s, col, (20, 10), 3)
    return s


def save_csv(data):
    name = f"Log_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    try:
        with open(name, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow([
                "Time_ms", "X", "Y", "Heading_deg", "Speed_px_s",
                "Error_px", "DR_Error_px", "EKF_Cov", "GDOP",
                "Visible_Beacons", "Beacon_IDs", "Event"
            ])
            w.writerows(data)
        return name
    except Exception:
        return None


def generate_base_path(margin=130, row_step=None, left_to_right=True):
    """Бустрофедонный маршрут (змейкой)."""
    path = []
    row_step = row_step or (IMPLEMENT_WIDTH + 10)
    num_rows = max(2, int((FIELD_AREA_H - 200) / row_step))
    for i in range(num_rows):
        y = margin + TAB_HEIGHT + i * row_step
        left_first = (i % 2 == 0) if left_to_right else (i % 2 != 0)
        if left_first:
            path.append((margin, y))
            path.append((FIELD_AREA_W - margin, y))
        else:
            path.append((FIELD_AREA_W - margin, y))
            path.append((margin, y))
    return path


def generate_obstacles_evenly(count, field_w, field_h, tab_h):
    """Равномерно размещает препятствия по полю через сетку ячеек."""
    obstacles = []
    cols = max(1, math.ceil(math.sqrt(count * field_w / field_h)))
    rows = max(1, math.ceil(count / cols))
    margin_x, margin_y = 250, 180
    cell_w = (field_w - 2 * margin_x) / cols
    cell_h = (field_h - 2 * margin_y) / rows
    positions = []
    for r in range(rows):
        for c in range(cols):
            cx = margin_x + c * cell_w + cell_w / 2
            cy = margin_y + tab_h + r * cell_h + cell_h / 2
            positions.append((
                int(cx + random.uniform(-cell_w * 0.2, cell_w * 0.2)),
                int(cy + random.uniform(-cell_h * 0.2, cell_h * 0.2))
            ))
    random.shuffle(positions)
    for i in range(min(count, len(positions))):
        t = random.choice(['rock', 'tree'])
        obstacles.append(Obstacle(positions[i][0], positions[i][1], t))
    return obstacles


def optimize_path_with_memory(base_path, memory_obs):
    """Искривляет маршрут на основе памяти об известных препятствиях."""
    if not isinstance(memory_obs, list):
        memory_obs = []
    new_path = []
    for i in range(len(base_path) - 1):
        p1 = base_path[i]
        p2 = base_path[i + 1]
        new_path.append(p1)
        for mem in memory_obs:
            mx, my, mr = mem['x'], mem['y'], mem['r']
            safe_dist = mr + (IMPLEMENT_WIDTH / 2) + 40
            if segment_circle_intersection(p1, p2, (mx, my), safe_dist):
                ux, uy = p2[0] - p1[0], p2[1] - p1[1]
                L = math.hypot(ux, uy)
                if L > 0:
                    ux, uy = ux / L, uy / L
                    if uy * (mx - p1[0]) - ux * (my - p1[1]) > 0:
                        nx, ny = -uy, ux
                    else:
                        nx, ny = uy, -ux
                    proj = (mx - p1[0]) * ux + (my - p1[1]) * uy
                    D = safe_dist
                    new_path.append(clamp_pt((p1[0] + ux * max(0, proj - mr - 70) + nx * D,
                                              p1[1] + uy * max(0, proj - mr - 70) + ny * D)))
                    new_path.append(clamp_pt((p1[0] + ux * proj + nx * D,
                                              p1[1] + uy * proj + ny * D)))
                    new_path.append(clamp_pt((p1[0] + ux * (proj + mr + 70) + nx * D,
                                              p1[1] + uy * (proj + mr + 70) + ny * D)))
    new_path.append(base_path[-1])
    return smooth_path(new_path, iterations=3)


def is_obstacle_already_known(mem_list, x, y, r, proximity=40):
    """
    УЛУЧШЕНИЕ: дедупликация препятствий в памяти на основе близости,
    а не точного совпадения координат.
    """
    for m in mem_list:
        if math.hypot(m['x'] - x, m['y'] - y) < proximity:
            return True
    return False


# --- UI ---
TAB_TITLES = ["СИМУЛЯТОР", "АНАЛИТИКА", "ПЛАН МИССИИ"]
TAB_W = 220


def get_tab_index_at(mx, my):
    if my >= TAB_HEIGHT or mx < 0 or mx >= TAB_W * len(TAB_TITLES):
        return None
    return int(mx // TAB_W)


def get_analysis_button_rect():
    return pygame.Rect(WINDOW_W - 380, WINDOW_H - 120, 300, 50)


def draw_card(screen, rect):
    pygame.draw.rect(screen, C_PANEL, rect, border_radius=8)
    pygame.draw.rect(screen, (38, 38, 45), rect, 1, border_radius=8)


def draw_section_label(screen, font, text, x, y):
    screen.blit(font.render(text, True, (100, 100, 110)), (x, y))


def sidebar_separator(screen, x, y):
    pygame.draw.line(screen, (50, 50, 55), (x, y), (x + SIDEBAR_W - 60, y), 1)


def draw_tabs(screen, font, current_tab):
    pygame.draw.rect(screen, (12, 12, 15), (0, 0, WINDOW_W, TAB_HEIGHT))
    for i, tab_name in enumerate(TAB_TITLES):
        rect = pygame.Rect(i * TAB_W, 0, TAB_W, TAB_HEIGHT)
        color = C_PANEL if i == current_tab else (12, 12, 15)
        pygame.draw.rect(screen, color, rect)
        if i == current_tab:
            pygame.draw.line(screen, C_ACCENT, (rect.left, TAB_HEIGHT - 1), (rect.right, TAB_HEIGHT - 1), 2)
        if i > 0:
            pygame.draw.line(screen, (22, 22, 28), (rect.left, 0), (rect.left, TAB_HEIGHT), 1)
        text = font.render(tab_name, True, C_TEXT_MAIN if i == current_tab else C_TEXT_DIM)
        screen.blit(text, text.get_rect(center=rect.center))
    pygame.draw.line(screen, (50, 50, 55), (0, TAB_HEIGHT), (WINDOW_W, TAB_HEIGHT), 1)


def draw_info_row(screen, font, x, y, label, value, val_color=C_TEXT_MAIN, right_x=None):
    screen.blit(font.render(label, True, C_TEXT_DIM), (x, y))
    val_surf = font.render(value, True, val_color)
    right_x = right_x or (x + SIDEBAR_W - 80)
    screen.blit(val_surf, (right_x - val_surf.get_width(), y))


def draw_live_chart(screen, rect, series_specs, thresholds=None):
    pygame.draw.rect(screen, (18, 18, 22), rect, border_radius=6)
    pygame.draw.rect(screen, (24, 24, 30), rect, 1, border_radius=6)
    chart_font = pygame.font.SysFont("Arial", 10)
    thresholds = thresholds or []
    values = []
    for data, _, _ in series_specs:
        values.extend(v for v in data if isinstance(v, (int, float)) and v < 999)
    values.extend(v for v, _, _ in thresholds)
    max_val = max(values) if values else 1.0
    max_val = max(max_val * 1.15, 1.0)

    for i in range(1, 4):
        y = rect.top + int(rect.h * i / 4)
        pygame.draw.line(screen, (34, 34, 40), (rect.left, y), (rect.right, y), 1)

    for value, color, label in thresholds:
        y = rect.bottom - int(min(value / max_val, 1.0) * rect.h)
        pygame.draw.line(screen, color, (rect.left, y), (rect.right, y), 1)
        label_surf = chart_font.render(label, True, color)
        screen.blit(label_surf, (rect.right - label_surf.get_width() - 4, y - 14))

    old_clip = screen.get_clip()
    screen.set_clip(rect)
    for data, color, filled in series_specs:
        clean = [v for v in data if isinstance(v, (int, float)) and v < 999]
        if len(clean) < 2:
            continue
        visible = clean[-max(2, rect.w):]
        denom = max(1, len(visible) - 1)
        pts = [
            (
                rect.left + int(i / denom * (rect.w - 1)),
                rect.bottom - int(min(max(val, 0) / max_val, 1.0) * (rect.h - 1))
            )
            for i, val in enumerate(visible)
        ]
        if filled:
            fill_surf = pygame.Surface((rect.w, rect.h), pygame.SRCALPHA)
            local_pts = [(x - rect.left, y - rect.top) for x, y in pts]
            pygame.draw.polygon(fill_surf, (*color, 18),
                                local_pts + [(local_pts[-1][0], rect.h), (local_pts[0][0], rect.h)])
            screen.blit(fill_surf, rect)
        pygame.draw.lines(screen, color, False, pts, 2)
    screen.set_clip(old_clip)


def draw_sim_sidebar(screen, font_sm, font_main, font_title, rob, ekf, err, vis, mode, rec, ha, state_str, lid,
                     avoid_cnt, mem_cnt, gdop=99.0, dr_err=0.0, route_points=0, ekf_trace=0.0):
    rect = pygame.Rect(FIELD_AREA_W, TAB_HEIGHT, SIDEBAR_W, WINDOW_H - TAB_HEIGHT)
    pygame.draw.rect(screen, C_PANEL, rect)
    pygame.draw.line(screen, (40, 40, 45), (FIELD_AREA_W, TAB_HEIGHT), (FIELD_AREA_W, WINDOW_H), 1)
    x, y = FIELD_AREA_W + 30, TAB_HEIGHT + 30

    screen.blit(font_title.render("НАВИГАЦИЯ", True, C_ACCENT), (x, y))
    y += 40
    sidebar_separator(screen, x, y)
    y += 22

    draw_section_label(screen, font_sm, "СИСТЕМНЫЙ СТАТУС", x, y)
    y += 28
    if state_str == "CRASH":
        st_col = C_ERR
        txt = "АВАРИЯ (СТОП)"
    elif state_str == "REVERSE":
        st_col = C_WARN
        txt = "РЕВЕРС (МАНЕВР)"
    else:
        st_col = C_GOOD if mode else C_TEXT_DIM
        txt = "АВТОПИЛОТ" if mode else "РУЧНОЕ УПР."
    draw_info_row(screen, font_main, x, y, "Управление:", txt, st_col)
    y += 30

    lid_front = lid.get_front_distance()
    lid_txt = "СВОБОДНО" if lid_front > 80 else ("ВНИМАНИЕ" if lid_front > 40 else "ОПАСНОСТЬ")
    draw_info_row(screen, font_main, x, y, "Статус радара:", lid_txt,
                  C_GOOD if lid_front > 80 else (C_WARN if lid_front > 40 else C_ERR))
    y += 42

    sidebar_separator(screen, x, y)
    y += 22
    draw_section_label(screen, font_sm, "НАВИГАЦИЯ И СЕТЬ", x, y)
    y += 28
    mesh_ok = any(len(b.neighbors) > 0 for b in vis)
    draw_info_row(screen, font_main, x, y, "Mesh-топология:",
                  "СТАБИЛЬНО" if mesh_ok else "ПОИСК УЗЛОВ",
                  C_ACCENT if mesh_ok else C_WARN)
    y += 30
    draw_info_row(screen, font_main, x, y, "Активные маяки:", f"{len(vis)} / 5", C_TEXT_MAIN)
    y += 30
    if gdop < 2.0:
        gdop_col = C_GOOD
        gdop_txt = f"{gdop:.2f} (ОТЛИЧНО)"
    elif gdop < 4.0:
        gdop_col = (180, 255, 0)
        gdop_txt = f"{gdop:.2f} (ХОРОШО)"
    elif gdop < 8.0:
        gdop_col = C_WARN
        gdop_txt = f"{gdop:.2f} (СРЕДНЕ)"
    else:
        gdop_col = C_ERR
        gdop_txt = f"{gdop:.1f} (ПЛОХО)"
    draw_info_row(screen, font_main, x, y, "GDOP:", gdop_txt, gdop_col)
    y += 42

    sidebar_separator(screen, x, y)
    y += 22
    draw_section_label(screen, font_sm, "МИССИЯ", x, y)
    y += 28
    draw_info_row(screen, font_main, x, y, "Тип маршрута:", "Покрытие поля", C_ACCENT)
    y += 30
    draw_info_row(screen, font_main, x, y, "Опорных точек:", str(route_points), C_TEXT_MAIN)
    y += 30
    draw_info_row(screen, font_main, x, y, "Обработано:", f"{ha:.4f} ГА", C_GOOD)
    y += 42

    sidebar_separator(screen, x, y)
    y += 22
    draw_section_label(screen, font_sm, "ТОЧНОСТЬ EKF", x, y)
    y += 28
    draw_info_row(screen, font_main, x, y, "Ошибка EKF:", f"{err:.1f} px",
                  C_ERR if err > 15 else C_GOOD)
    y += 25
    bar_w = SIDEBAR_W - 80
    pygame.draw.rect(screen, (40, 40, 45), (x, y, bar_w, 5))
    pygame.draw.rect(screen, (0, 150, 255), (x, y, int(min(err, 60) / 60 * bar_w), 5))
    y += 25
    draw_info_row(screen, font_main, x, y, "trace(P):", f"{ekf_trace:.3f}", C_TEXT_MAIN)
    y += 28

    sidebar_separator(screen, x, y)
    y += 22
    draw_section_label(screen, font_sm, "ГОРЯЧИЕ КЛАВИШИ", x, y)
    y += 30
    cmds = [
        ("SPACE", "Авто/Ручной"),
        ("E", "План миссии"),
        ("BACKSPACE", "Сброс симуляции"),
        ("S", "Запись лога"),
        ("H", "Тепловая карта"),
        ("L", "Лидар вкл/выкл"),
        ("P", "Показать/скрыть маршрут"),
        ("V", "Показать/скрыть покрытие"),
    ]
    for k, d in cmds:
        screen.blit(font_sm.render(k, True, C_ACCENT), (x, y))
        screen.blit(font_sm.render(d, True, C_TEXT_DIM), (x + 100, y))
        y += 22


    if state_str == "CRASH":
        y += 20
        al = font_main.render("УВЕДОМЛЕНИЕ ОТПРАВЛЕНО В TELEGRAM", True, C_ERR)
        screen.blit(al, al.get_rect(center=(x + (SIDEBAR_W - 60) // 2, y)))


def path_length(points):
    if len(points) < 2:
        return 0
    return sum(math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1])
               for i in range(1, len(points)))


def draw_field_grid(screen, rect):
    pygame.draw.rect(screen, C_BG, rect)
    for gx in range(0, FIELD_AREA_W + 1, 200):
        pygame.draw.line(screen, C_GRID, (gx, TAB_HEIGHT), (gx, WINDOW_H), 1)
    for gy in range(TAB_HEIGHT, WINDOW_H + 1, 200):
        pygame.draw.line(screen, C_GRID, (0, gy), (FIELD_AREA_W, gy), 1)


def draw_mission_screen(screen, font_sm, font_main, font_title, obs, preview_path,
                        mission_profile, mission_margin, mission_step, mission_left_to_right):
    field_rect = pygame.Rect(0, TAB_HEIGHT, FIELD_AREA_W, FIELD_AREA_H)
    screen.fill(C_BG)
    screen.set_clip(field_rect)
    draw_field_grid(screen, field_rect)

    if len(preview_path) > 1:
        pygame.draw.lines(screen, C_ACCENT, False, preview_path, 2)
        dot_step = max(1, len(preview_path) // 110)
        for p in preview_path[::dot_step]:
            pygame.draw.circle(screen, C_ACCENT, (int(p[0]), int(p[1])), 3)

    for o in obs:
        o.draw(screen)
    screen.set_clip(None)

    rect = pygame.Rect(FIELD_AREA_W, TAB_HEIGHT, SIDEBAR_W, WINDOW_H - TAB_HEIGHT)
    pygame.draw.rect(screen, C_PANEL, rect)
    pygame.draw.line(screen, (40, 40, 45), (FIELD_AREA_W, TAB_HEIGHT), (FIELD_AREA_W, WINDOW_H), 1)
    x, y = FIELD_AREA_W + 30, TAB_HEIGHT + 30

    screen.blit(font_title.render("ПЛАН МИССИИ", True, C_ACCENT), (x, y))
    y += 38
    screen.blit(font_sm.render("Подготовьте маршрут до запуска автопилота.", True, C_TEXT_DIM), (x, y))
    y += 28
    sidebar_separator(screen, x, y)
    y += 22

    draw_section_label(screen, font_sm, "ПРОФИЛЬ", x, y)
    y += 28
    draw_info_row(screen, font_main, x, y, "Режим:", mission_profile, C_ACCENT)
    y += 30
    draw_info_row(screen, font_main, x, y, "Отступ поля:", f"{mission_margin} px")
    y += 30
    draw_info_row(screen, font_main, x, y, "Шаг прохода:", f"{mission_step} px")
    y += 30
    first_pass = "Слева направо" if mission_left_to_right else "Справа налево"
    draw_info_row(screen, font_main, x, y, "Первый проход:", first_pass)
    y += 42

    sidebar_separator(screen, x, y)
    y += 22
    draw_section_label(screen, font_sm, "ОЦЕНКА", x, y)
    y += 28
    draw_info_row(screen, font_main, x, y, "Точек маршрута:", str(len(preview_path)))
    y += 30
    draw_info_row(screen, font_main, x, y, "Длина пути:", f"{int(path_length(preview_path))} px")
    y += 30
    draw_info_row(screen, font_main, x, y, "Препятствий на карте:", str(len(obs)))
    y += 42

    sidebar_separator(screen, x, y)
    y += 22
    draw_section_label(screen, font_sm, "УПРАВЛЕНИЕ", x, y)
    y += 30
    cmds = [
        ("M", "Сменить профиль миссии"),
        ("ENTER / E", "Применить маршрут"),
        ("ESC", "Вернуться без изменений"),
        ("UP / DOWN", "Менять отступ поля"),
        ("LEFT / RIGHT", "Менять шаг прохода"),
        ("R", "Сменить стартовую сторону"),
    ]
    for k, d in cmds:
        screen.blit(font_sm.render(k, True, C_ACCENT), (x, y))
        screen.blit(font_sm.render(d, True, C_TEXT_DIM), (x + 120, y))
        y += 24

    draw_tabs(screen, font_title, 2)


# --- MAIN ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("AgroNav: Professional Research Environment")
    clock = pygame.time.Clock()

    font_sm = pygame.font.SysFont("Arial", 12)
    font_main = pygame.font.SysFont("Arial", 14)
    font_title = pygame.font.SysFont("Arial", 18, bold=True)

    dirt = create_dirt(FIELD_AREA_W, FIELD_AREA_H)
    tr_ok, tr_bad = create_tractor(False), create_tractor(True)
    bc_on = create_beacon(C_ACCENT)
    bc_off = create_beacon((80, 80, 85))
    beacon_err = create_beacon(C_WARN)

    f_rect = pygame.Rect(0, TAB_HEIGHT, FIELD_AREA_W, FIELD_AREA_H)
    cov = CoverageManager(FIELD_AREA_W, FIELD_AREA_H)
    lid = SectorLidar(radius=220)

    obs_count = CONFIG["simulation"].get("obstacles_count", 3)
    obs = generate_obstacles_evenly(obs_count, FIELD_AREA_W, FIELD_AREA_H, TAB_HEIGHT)
    bcs = [
        Beacon(1, 150,                    150 + TAB_HEIGHT,              800, None, fail_prob=0.003),
        Beacon(2, FIELD_AREA_W - 150,     150 + TAB_HEIGHT,              800, None, fail_prob=0.003),
        Beacon(3, 150,                    FIELD_AREA_H + TAB_HEIGHT - 150, 800, None, fail_prob=0.003),
        Beacon(4, FIELD_AREA_W - 150,     FIELD_AREA_H + TAB_HEIGHT - 150, 800, None, fail_prob=0.003),
        Beacon(5, FIELD_AREA_W // 2,      FIELD_AREA_H // 2 + TAB_HEIGHT, 600, None, fail_prob=0.001),
    ]

    calculate_mesh_network(bcs)
    heatmap_surface = generate_heatmap(FIELD_AREA_W, FIELD_AREA_H, bcs)

    tractor_speed = CONFIG["simulation"].get("tractor_speed", 160)
    rob = Robot(200, 200 + TAB_HEIGHT, 0)
    ekf = EKF(200, 200 + TAB_HEIGHT, 0)

    # Dead Reckoning трекер — копия позиции, обновляется ТОЛЬКО по одометрии,
    # без каких-либо поправок от маяков. Показывает куда привела бы одна одометрия.
    dr = {'x': 200.0, 'y': float(200 + TAB_HEIGHT), 'theta': 0.0}
    dr_err_history = []
    dr_path = []   # траектория DR для отрисовки

    ai_memory = load_ai_memory()
    mission_profiles = [
        ("Покрытие поля", 130, IMPLEMENT_WIDTH + 10),
        ("Точный проход", 100, max(45, IMPLEMENT_WIDTH - 5)),
        ("Широкий проход", 160, IMPLEMENT_WIDTH + 30),
    ]
    mission_profile_idx = 0
    mission_profile, mission_margin, mission_step = mission_profiles[mission_profile_idx]
    mission_left_to_right = True

    if ai_memory.get("learned_path"):
        path = smooth_path(ai_memory["learned_path"])
        using_learned_path = True
    else:
        base_p = generate_base_path(mission_margin, mission_step, mission_left_to_right)
        path = optimize_path_with_memory(base_p, ai_memory.get("obstacles", []))
        using_learned_path = False

    app_state = "SIM"
    actual_driven_path = []
    tractor_state = "FORWARD"
    reverse_timer = 0
    wp = 0
    auto = False
    rec = False
    heat = False
    current_gdop = 99.0
    log_event = ""   # текущее событие для записи в лог этого кадра
    obstacles_avoided = 0
    show_route = True   # видимость синей линии маршрута
    show_coverage = True   # видимость закрашенной зоны обработанного поля
    log = []
    err_history = []
    gdop_history = []
    cov_history = []
    msgt = 0
    msg = ""

    def build_mission_path():
        base_path = generate_base_path(mission_margin, mission_step, mission_left_to_right)
        return optimize_path_with_memory(base_path, ai_memory.get("obstacles", []))

    def apply_mission_path():
        nonlocal path, wp, using_learned_path, app_state, auto
        path = build_mission_path()
        wp = 0
        using_learned_path = False
        app_state = "SIM"
        auto = False

    # ИСПРАВЛЕНИЕ: surface с альфа-каналом для рисования полупрозрачных кругов памяти
    mem_overlay = pygame.Surface((FIELD_AREA_W, FIELD_AREA_H), pygame.SRCALPHA)

    run = True
    while run:
        dt = clock.tick(FPS) / 1000.0
        events = pygame.event.get()

        for e in events:
            if e.type == pygame.QUIT:
                run = False

            if e.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                tab_idx = get_tab_index_at(mx, my)
                if tab_idx is not None:
                    if tab_idx == 0:
                        app_state = "SIM"
                    elif tab_idx == 1:
                        app_state = "ANALYZE"
                    elif tab_idx == 2:
                        app_state = "MISSION"
                        auto = False

                elif app_state == "ANALYZE":
                    btn_rect = get_analysis_button_rect()
                    if btn_rect.collidepoint(mx, my):
                        open_file_and_analyze()

                elif app_state == "SIM":
                    if f_rect.collidepoint(mx, my):
                        for b in bcs:
                            if math.hypot(mx - b.x, my - b.y) < 30:
                                b.active = not b.active
                                calculate_mesh_network(bcs)
                                heatmap_surface = generate_heatmap(FIELD_AREA_W, FIELD_AREA_H, bcs)

            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_v:
                    show_coverage = not show_coverage
                if e.key == pygame.K_TAB:
                    states = ["SIM", "ANALYZE", "MISSION"]
                    app_state = states[(states.index(app_state) + 1) % len(states)]
                    if app_state == "MISSION":
                        auto = False

                if app_state == "MISSION":
                    if e.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_e):
                        apply_mission_path()
                    elif e.key == pygame.K_ESCAPE:
                        app_state = "SIM"
                    elif e.key == pygame.K_UP:
                        mission_margin = min(240, mission_margin + 10)
                    elif e.key == pygame.K_DOWN:
                        mission_margin = max(60, mission_margin - 10)
                    elif e.key == pygame.K_RIGHT:
                        mission_step = min(140, mission_step + 5)
                    elif e.key == pygame.K_LEFT:
                        mission_step = max(45, mission_step - 5)
                    elif e.key == pygame.K_r:
                        mission_left_to_right = not mission_left_to_right
                    elif e.key == pygame.K_m:
                        mission_profile_idx = (mission_profile_idx + 1) % len(mission_profiles)
                        mission_profile, mission_margin, mission_step = mission_profiles[mission_profile_idx]

                elif e.key == pygame.K_e and app_state == "SIM":
                    app_state = "MISSION"
                    auto = False

                if app_state == "SIM":
                    if e.key == pygame.K_t:
                        if len(actual_driven_path) > 10:
                            ai_memory["learned_path"] = actual_driven_path[::20]
                            save_ai_memory(ai_memory)
                            msg = "ОПЫТ СОХРАНЕН"
                            msgt = 120

                    if e.key == pygame.K_c:
                        ai_memory = {"obstacles": [], "learned_path": []}
                        save_ai_memory(ai_memory)
                        path = build_mission_path()
                        wp = 0
                        using_learned_path = False
                        msg = "ПАМЯТЬ ОЧИЩЕНА"
                        msgt = 120

                    if e.key == pygame.K_BACKSPACE:
                        save_ai_memory(ai_memory)
                        rob.x, rob.y, rob.theta = 200, 200 + TAB_HEIGHT, 0
                        ekf.x = np.array([200, 200 + TAB_HEIGHT, 0], dtype=float)
                        ekf.P = np.eye(3) * 10.0
                        dr['x'], dr['y'], dr['theta'] = 200, 200 + TAB_HEIGHT, 0
                        dr_err_history.clear()
                        dr_path.clear()
                        wp = 0
                        auto = False
                        tractor_state = "FORWARD"
                        obstacles_avoided = 0
                        actual_driven_path = []

                        if ai_memory.get("learned_path"):
                            path = smooth_path(ai_memory["learned_path"])
                            using_learned_path = True
                        else:
                            path = build_mission_path()
                            using_learned_path = False

                        cov.surf.fill((0, 0, 0, 0))
                        cov.cells.clear()
                        err_history.clear()
                        gdop_history.clear()
                        cov_history.clear()
                        log.clear()
                        for o in obs:
                            o.known = False
                        obs = generate_obstacles_evenly(obs_count, FIELD_AREA_W, FIELD_AREA_H, TAB_HEIGHT)
                        msg = "СИСТЕМА ПЕРЕЗАГРУЖЕНА"
                        msgt = 120

                    if e.key == pygame.K_SPACE and tractor_state != "CRASH":
                        auto = not auto
                    if e.key == pygame.K_l:
                        lid.visible = not lid.visible
                    if e.key == pygame.K_h:
                        heat = not heat
                    if e.key == pygame.K_p:
                        show_route = not show_route
                    if e.key == pygame.K_s:
                        if not rec:
                            rec = True
                            log = []
                        else:
                            n = save_csv(log)
                            rec = False
                            if n:
                                msg = f"ЛОГ СОХРАНЁН: {n}"
                                msgt = 180

        # --- ЛОГИКА СИМУЛЯЦИИ ---
        if app_state == "ANALYZE":
            draw_analysis_screen(screen, font_title, font_main, font_sm,
                                 err_history, dr_err_history, gdop_history, cov_history,
                                 log, rec, obstacles_avoided)
            draw_tabs(screen, font_title, 1)

        elif app_state == "MISSION":
            mission_preview = build_mission_path()
            draw_mission_screen(screen, font_sm, font_main, font_title, obs, mission_preview,
                                mission_profile, mission_margin, mission_step, mission_left_to_right)

        elif app_state == "SIM":
            # 1. Сброс флага «известности» только когда препятствие позади
            fwd_x = math.cos(rob.theta)
            fwd_y = math.sin(rob.theta)
            for o in obs:
                if o.known:
                    dot = (o.x - rob.x) * fwd_x + (o.y - rob.y) * fwd_y
                    if dot < -80:
                        o.known = False
                        o.tg_alerted = False

            # 2. Лидар
            log_event = ""   # сбрасываем событие каждый кадр
            lid.scan(rob.x, rob.y, rob.theta, obs)
            front_dist = lid.get_front_distance()

            # 3. Краш-тест
            if tractor_state != "CRASH":
                rr = pygame.Rect(rob.x - 25, rob.y - 25, 50, 50)
                for o in obs:
                    if rr.colliderect(o.rect):
                        tractor_state = "CRASH"
                        auto = False
                        trigger_alert("АВАРИЯ! Трактор столкнулся с препятствием.")
                        msg = "СИСТЕМА ПОВРЕЖДЕНА"
                        msgt = 120

            # 4. Автоматическое предотвращение столкновений
            if auto and tractor_state == "FORWARD":
                if front_dist < STOP_DISTANCE:
                    tractor_state = "REVERSE"
                    reverse_timer = 45
                    log_event = "REVERSE"
                    msg = "ЭКСТРЕННЫЙ РЕВЕРС"
                    msgt = 45
                    for o in obs:
                        if math.hypot(o.x - rob.x, o.y - rob.y) < 100:
                            obs_list = ai_memory.get("obstacles", [])
                            if not is_obstacle_already_known(obs_list, o.x, o.y, o.radius):
                                obs_list.append({'x': o.x, 'y': o.y, 'r': o.radius})
                                ai_memory["obstacles"] = obs_list
                                trigger_alert("Новый объект добавлен в базу знаний.")
                                save_ai_memory(ai_memory)

                elif not using_learned_path and wp < len(path):
                    target = path[wp]
                    for o in obs:
                        if not o.known and math.hypot(o.x - rob.x, o.y - rob.y) < AVOID_DISTANCE + o.radius:
                            angle_diff = abs(
                                (math.atan2(o.y - rob.y, o.x - rob.x) - rob.theta + math.pi) % (2 * math.pi) - math.pi
                            )
                            if angle_diff < math.radians(70):
                                safe_R = o.radius + (IMPLEMENT_WIDTH // 2) + 30
                                if segment_circle_intersection((rob.x, rob.y), target, (o.x, o.y), safe_R):
                                    o.known = True
                                    obstacles_avoided += 1
                                    log_event = "AVOIDANCE"
                                    msg = "МАНЕВР ОБЪЕЗДА"
                                    msgt = 120
                                    if not o.tg_alerted:
                                        trigger_alert("⚠️ Препятствие по курсу — выполняю объезд!")
                                        o.tg_alerted = True

                                    ux, uy = target[0] - rob.x, target[1] - rob.y
                                    L = math.hypot(ux, uy)
                                    if L > 0:
                                        ux, uy = ux / L, uy / L
                                        if ux * (o.y - rob.y) - uy * (o.x - rob.x) < 0:
                                            nx, ny = -uy, ux
                                        else:
                                            nx, ny = uy, -ux

                                        proj = (o.x - rob.x) * ux + (o.y - rob.y) * uy
                                        D = safe_R + 8
                                        w1 = clamp_pt((rob.x + ux * max(10, proj - 60) + nx * D,
                                                       rob.y + uy * max(10, proj - 60) + ny * D))
                                        w2 = clamp_pt((rob.x + ux * proj + nx * D,
                                                       rob.y + uy * proj + ny * D))
                                        w3 = clamp_pt((rob.x + ux * (proj + o.radius + 40) + nx * D,
                                                       rob.y + uy * (proj + o.radius + 40) + ny * D))

                                        pts_to_del = [i for i in range(wp, len(path))
                                                      if math.hypot(path[i][0] - o.x,
                                                                    path[i][1] - o.y) < safe_R]
                                        for i in reversed(pts_to_del):
                                            path.pop(i)
                                        path.insert(wp, w3)
                                        path.insert(wp, w2)
                                        path.insert(wp, w1)
                                        # smooth_path убран — он срезал углы обратно к препятствию
                                        break

            # 5. Движение
            v, w = 0, 0
            if tractor_state == "REVERSE":
                v = -50
                w = 0
                reverse_timer -= 1
                if reverse_timer <= 0:
                    tractor_state = "FORWARD"
            elif tractor_state == "FORWARD" and auto:
                if wp < len(path):
                    target = path[wp]
                    dist = math.hypot(target[0] - rob.x, target[1] - rob.y)
                    while dist < 40 and wp < len(path) - 1:
                        wp += 1
                        target = path[wp]
                        dist = math.hypot(target[0] - rob.x, target[1] - rob.y)

                    diff = (math.atan2(target[1] - rob.y, target[0] - rob.x) - rob.theta + math.pi) % (
                        2 * math.pi) - math.pi
                    if abs(diff) > math.radians(45):
                        v = tractor_speed * 0.3
                        w = math.copysign(2.5, diff)
                    elif abs(diff) > math.radians(10):
                        v = tractor_speed * 0.7
                        w = diff * 4.0
                    else:
                        v = tractor_speed
                        w = diff * 3.5
                    actual_driven_path.append((rob.x, rob.y))
            elif not auto and tractor_state != "CRASH":
                k = pygame.key.get_pressed()
                v = 160 if k[pygame.K_UP] else (-80 if k[pygame.K_DOWN] else 0)
                w = 3.0 if k[pygame.K_RIGHT] else (-3.0 if k[pygame.K_LEFT] else 0)

            if tractor_state != "CRASH":
                rob.move(v, w, dt)
                ov, ow = rob.get_odometry(v, w, dt)
                ekf.predict(ov, ow, dt)

                # DR: интегрируем ТЕ ЖЕ зашумлённые показания одометрии,
                # но без update() от маяков — чистая интеграция без коррекции
                dr['theta'] += ow * dt
                dr['theta'] = (dr['theta'] + math.pi) % (2 * math.pi) - math.pi
                dr['x'] += ov * math.cos(dr['theta']) * dt
                dr['y'] += ov * math.sin(dr['theta']) * dt
                dr_path.append((dr['x'], dr['y']))
                if len(dr_path) > 4000:
                    dr_path.pop(0)

                if v > 0:
                    cov.paint(rob.x, rob.y - TAB_HEIGHT, rob.theta)

            vis = []
            mesh_nodes = set()

            # Шаг 1: собираем все доступные маяки, детектируем отказы
            candidates = []
            for b in bcs:
                in_range = math.hypot(rob.x - b.x, rob.y - b.y) <= b.radius
                available = b.is_available(rob.x, rob.y)
                # Отказ = маяк в зоне видимости, активен, но не отвечает (fail_timer)
                if in_range and b.active and not available:
                    log_event = "BEACON_FAIL"
                if available:
                    mesh_connected = any(n.active for n in b.neighbors)
                    if mesh_connected:
                        mesh_nodes.add(b)
                        for n in b.neighbors:
                            if n.active:
                                mesh_nodes.add(n)
                    noise_std = b.get_noise_std(rob.x, rob.y, mesh_connected)
                    candidates.append((b, noise_std))

            # Шаг 2: выбор лучших маяков по GDOP
            # Если кандидатов > 3 — выбираем тройку с наилучшей геометрией.
            # Так работает реальный GPS/UWB: выбирает спутники с мин. PDOP.
            if len(candidates) > 3:
                best_set = candidates
                best_gdop = 99.0
                # Перебираем все тройки — для 5 маяков это всего C(5,3)=10 итераций
                from itertools import combinations
                for combo in combinations(candidates, 3):
                    gdop = compute_gdop(rob.x, rob.y, [c[0] for c in combo])
                    if gdop < best_gdop:
                        best_gdop = gdop
                        best_set = list(combo)
                selected = best_set
            else:
                selected = candidates

            # Шаг 3: EKF update с адаптивной матрицей R для каждого маяка
            for b, noise_std in selected:
                z = b.measure_distance(rob.x, rob.y, noise_std)
                ekf.update(z, (b.x, b.y), noise_std)
                vis.append(b)

            # Текущий GDOP для сайдбара
            current_gdop = compute_gdop(rob.x, rob.y, [b for b, _ in selected]) if selected else 99.0

            err    = math.hypot(rob.x - ekf.x[0], rob.y - ekf.x[1])
            dr_err = math.hypot(rob.x - dr['x'],   rob.y - dr['y'])
            ekf_cov = float(np.trace(ekf.P[:2, :2]))
            err_history.append(err)
            dr_err_history.append(dr_err)
            gdop_history.append(current_gdop)
            cov_history.append(ekf_cov)
            if len(err_history) > FIELD_AREA_W:
                err_history.pop(0)
            if len(dr_err_history) > FIELD_AREA_W:
                dr_err_history.pop(0)
            if len(gdop_history) > FIELD_AREA_W:
                gdop_history.pop(0)
            if len(cov_history) > FIELD_AREA_W:
                cov_history.pop(0)
            if rec:
                beacon_ids = ";".join(str(b.id) for b in vis) if vis else ""
                heading_deg = round(math.degrees(rob.theta) % 360, 1)
                log.append([
                    pygame.time.get_ticks(),
                    round(rob.x, 2), round(rob.y, 2),
                    heading_deg,
                    round(abs(v), 1),
                    round(err, 3),
                    round(dr_err, 3),
                    round(ekf_cov, 4),
                    round(current_gdop, 3),
                    len(vis),
                    beacon_ids,
                    log_event
                ])

            # --- ОТРИСОВКА ---
            screen.set_clip(f_rect)
            screen.fill(C_BG, f_rect)
            if heat:
                screen.blit(heatmap_surface, (0, TAB_HEIGHT))
            else:
                screen.blit(dirt, (0, TAB_HEIGHT))
                if show_coverage:
                    cov.draw(screen)

            for o in obs:
                o.draw(screen)

            # ИСПРАВЛЕНИЕ: полупрозрачные круги памяти через SRCALPHA-поверхность
            mem_overlay.fill((0, 0, 0, 0))
            for m in ai_memory.get("obstacles", []):
                pygame.draw.circle(mem_overlay, (255, 50, 50, 60),
                                   (int(m['x']), int(m['y']) - TAB_HEIGHT), int(m['r'] + 10), 2)
            screen.blit(mem_overlay, (0, TAB_HEIGHT))

            for gx in range(0, FIELD_AREA_W, 200):
                pygame.draw.line(screen, C_GRID, (gx, TAB_HEIGHT), (gx, WINDOW_H), 1)
            for gy in range(TAB_HEIGHT, WINDOW_H, 200):
                pygame.draw.line(screen, C_GRID, (0, gy), (FIELD_AREA_W, gy), 1)

            path_color = C_GOLD if using_learned_path else (0, 150, 255)
            if show_route and len(path[wp:]) > 1:
                pygame.draw.lines(screen, path_color, False, path[wp:], 2)

            drawn = set()
            for b in bcs:
                if not b.active:
                    continue
                for n in b.neighbors:
                    if not n.active:
                        continue
                    link_id = frozenset([b.id, n.id])
                    if link_id in drawn:
                        continue
                    drawn.add(link_id)
                    active_link = (b in mesh_nodes and n in mesh_nodes)
                    c = (0, 180, 200) if active_link else (60, 60, 70)
                    wd = 2 if active_link else 1
                    pygame.draw.line(screen, c, (b.x, b.y), (n.x, n.y), wd)

            for b in bcs:
                im = bc_off
                if b.active:
                    im = bc_on if b in vis else beacon_err
                screen.blit(im, im.get_rect(center=(b.x, b.y)))

            # Эллипс ковариации EKF — оставляем, он небольшой и полезен
            try:
                ew, eh, ea = get_covariance_ellipse(ekf.P)
                s = pygame.Surface((max(1, int(ew)), max(1, int(eh))), pygame.SRCALPHA)
                pygame.draw.ellipse(s, (0, 200, 255, 20), (0, 0, max(1, int(ew)), max(1, int(eh))))
                pygame.draw.ellipse(s, (0, 200, 255, 60), (0, 0, max(1, int(ew)), max(1, int(eh))), 1)
                rs = pygame.transform.rotate(s, -ea)
                screen.blit(rs, rs.get_rect(center=(int(ekf.x[0]), int(ekf.x[1]))))
            except Exception:
                pass

            iox = rob.x - math.cos(rob.theta) * IMPLEMENT_OFFSET
            ioy = rob.y - math.sin(rob.theta) * IMPLEMENT_OFFSET
            ilx = iox + math.sin(rob.theta) * (IMPLEMENT_WIDTH / 2)
            ily = ioy - math.cos(rob.theta) * (IMPLEMENT_WIDTH / 2)
            irx = iox - math.sin(rob.theta) * (IMPLEMENT_WIDTH / 2)
            iry = ioy + math.cos(rob.theta) * (IMPLEMENT_WIDTH / 2)
            pygame.draw.line(screen, (150, 150, 160), (ilx, ily), (irx, iry), 4)

            lid.draw(screen, rob.x, rob.y, rob.theta)
            rt = pygame.transform.rotate(
                tr_bad if tractor_state == "CRASH" else tr_ok,
                -math.degrees(rob.theta) - 90
            )
            screen.blit(rt, rt.get_rect(center=(rob.x, rob.y)))
            screen.set_clip(None)

            draw_sim_sidebar(screen, font_sm, font_main, font_title, rob, ekf, err, vis, auto, rec,
                             cov.get_hectares(), tractor_state, lid,
                             avoid_cnt=obstacles_avoided,
                             mem_cnt=len(ai_memory.get("obstacles", [])),
                             gdop=current_gdop, dr_err=dr_err,
                             route_points=len(path), ekf_trace=ekf_cov)
            draw_tabs(screen, font_title, 0)

            if msgt > 0:
                s = font_main.render(msg, True, C_BG)
                rect_msg = pygame.Rect(0, 0, max(300, s.get_width() + 30), 40)
                rect_msg.center = (FIELD_AREA_W // 2, TAB_HEIGHT + 35)
                pygame.draw.rect(screen, C_WARN if any(w in msg for w in ["РЕВЕРС", "АВАРИЯ", "ПОВРЕЖДЕНА"]) else C_GOOD,
                                 rect_msg, border_radius=4)
                screen.blit(s, s.get_rect(center=rect_msg.center))
                msgt -= 1

        pygame.display.flip()

    save_ai_memory(ai_memory)
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
