import math
import json
import os
import requests
import threading
import csv
import datetime
import random


# --- ЗАГРУЗКА КОНФИГУРАЦИИ ---
def load_config():
    default_conf = {
        "window": {"width": 1920, "height": 1080, "fps": 60},
        "telegram": {"enabled": False, "token": "", "chat_id": ""},
        "simulation": {"beacon_range": 800, "obstacles_count": 7, "tractor_speed": 160, "tractor_width_px": 60,
                       "tractor_offset_px": 30},
        "lidar": {"range": 350, "fov": 360, "stop_distance": 50}
    }
    if not os.path.exists("settings.json"): return default_conf
    try:
        with open("settings.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return default_conf


CONFIG = load_config()

# --- ГЛОБАЛЬНЫЕ КОНСТАНТЫ ---
WINDOW_W = CONFIG["window"]["width"]
WINDOW_H = CONFIG["window"]["height"]
FPS = CONFIG["window"]["fps"]

TAB_HEIGHT = 40
SIDEBAR_W = 450
GRAPH_H = 220
FIELD_AREA_W = WINDOW_W - SIDEBAR_W
FIELD_AREA_H = WINDOW_H - GRAPH_H - TAB_HEIGHT

BEACON_COMMS_RANGE = CONFIG["simulation"]["beacon_range"]
IMPLEMENT_WIDTH = CONFIG["simulation"]["tractor_width_px"]
IMPLEMENT_OFFSET = CONFIG["simulation"]["tractor_offset_px"]
GRID_CELL_SIZE = 5

LIDAR_RANGE = CONFIG["lidar"]["range"]
LIDAR_FOV = 360
LIDAR_RAYS = 16
STOP_DISTANCE = 45
AVOID_DISTANCE = 250

# Настройки Телеграма
TG_BOT_TOKEN = "ВСТАВЬ_СЮДА_ТОКЕН"
TG_CHAT_ID = "ВСТАВЬ_СЮДА_ID"
TG_ENABLED = True

# --- ЦВЕТОВАЯ ПАЛИТРА ---
C_BG = (18, 18, 22);
C_PANEL = (25, 25, 30);
C_GRID = (40, 45, 50)
C_TEXT_MAIN = (240, 240, 245);
C_TEXT_DIM = (140, 140, 150);
C_ACCENT = (0, 150, 255)
C_WARN = (255, 170, 0);
C_ERR = (255, 60, 80);
C_GOOD = (40, 200, 100);
C_COVERAGE = (0, 180, 120)
C_GOLD = (255, 215, 0)

# --- ПАМЯТЬ ИИ ---
MEMORY_FILE = "ai_memory.json"


def load_ai_memory():
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, 'r') as f:
                data = json.load(f)
                if isinstance(data, list): return {"obstacles": data, "learned_path": []}
                return data
        except:
            pass
    return {"obstacles": [], "learned_path": []}


def save_ai_memory(memory_dict):
    with open(MEMORY_FILE, 'w') as f: json.dump(memory_dict, f)


# --- TELEGRAM ---
def send_telegram_alert_thread(message):
    if not TG_ENABLED or len(TG_BOT_TOKEN) < 10: return
    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
    data = {"chat_id": TG_CHAT_ID, "text": f"🤖 AgroNav System:\n{message}"}
    try:
        requests.post(url, data=data, timeout=5)
    except:
        pass


def trigger_alert(reason):
    threading.Thread(target=send_telegram_alert_thread, args=(reason,)).start()


# --- МАТЕМАТИКА ---
def ray_circle_intersection(ro, rd, cc, cr):
    oc = (cc[0] - ro[0], cc[1] - ro[1])
    tc = oc[0] * rd[0] + oc[1] * rd[1]
    if tc < 0: return None
    d2 = (oc[0] ** 2 + oc[1] ** 2) - tc ** 2
    if d2 > cr ** 2: return None
    return tc - math.sqrt(cr ** 2 - d2)


def segment_circle_intersection(A, B, C, R):
    Ax, Ay = A;
    Bx, By = B;
    Cx, Cy = C
    Dx, Dy = Bx - Ax, By - Ay;
    Fx, Fy = Cx - Ax, Cy - Ay
    AB_len_sq = Dx * Dx + Dy * Dy
    if AB_len_sq == 0: return math.hypot(Fx, Fy) <= R
    t = max(0, min(1, (Fx * Dx + Fy * Dy) / AB_len_sq))
    return math.hypot(Cx - (Ax + t * Dx), Cy - (Ay + t * Dy)) <= R


def clamp_pt(p):
    x = max(40, min(FIELD_AREA_W - 40, p[0]))
    y = max(TAB_HEIGHT + 40, min(FIELD_AREA_H + TAB_HEIGHT - 40, p[1]))
    return (x, y)


def smooth_path(path, iterations=3):
    if len(path) < 3: return path
    for _ in range(iterations):
        new_path = [path[0]]
        for i in range(len(path) - 1):
            p0 = path[i];
            p1 = path[i + 1]
            new_path.append((p0[0] * 0.75 + p1[0] * 0.25, p0[1] * 0.75 + p1[1] * 0.25))
            new_path.append((p0[0] * 0.25 + p1[0] * 0.75, p0[1] * 0.25 + p1[1] * 0.75))
        new_path.append(path[-1])
        path = new_path
    return path


def save_csv(data):
    name = f"Log_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    try:
        with open(name, 'w', newline='') as f:
            w = csv.writer(f);
            w.writerow(["Time_ms", "X", "Y", "Error_px", "Visible_Beacons"])
            w.writerows(data)
        return name
    except:
        return None


def generate_base_path():
    path = []
    margin = 120
    for i in range(int((FIELD_AREA_H - 200) / (IMPLEMENT_WIDTH + 10))):
        y = margin + TAB_HEIGHT + i * (IMPLEMENT_WIDTH + 10)
        path.append((margin, y));
        path.append((FIELD_AREA_W - margin, y)) if i % 2 == 0 else path.append((margin, y))
    return path


def optimize_path_with_memory(base_path, memory_obs):
    """ Искривляет маршрут на основе памяти (AI Memory) """
    new_path = []
    for i in range(len(base_path) - 1):
        p1 = base_path[i];
        p2 = base_path[i + 1];
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
                    new_path.append(clamp_pt(
                        (p1[0] + ux * max(0, proj - mr - 60) + nx * D, p1[1] + uy * max(0, proj - mr - 60) + ny * D)))
                    new_path.append(clamp_pt((p1[0] + ux * proj + nx * D, p1[1] + uy * proj + ny * D)))
                    new_path.append(
                        clamp_pt((p1[0] + ux * (proj + mr + 60) + nx * D, p1[1] + uy * (proj + mr + 60) + ny * D)))

    new_path.append(base_path[-1])
    return smooth_path(new_path, iterations=3)