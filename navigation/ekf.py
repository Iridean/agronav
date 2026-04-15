import numpy as np
import math


class EKF:
    def __init__(self, x0, y0, theta0):
        self.x = np.array([x0, y0, theta0], dtype=float)
        self.P = np.eye(3) * 10.0
        # Шум процесса (одометрия): x, y, theta
        self.Q = np.diag([0.1, 0.1, 0.01])
        # Базовая матрица шума измерений — теперь используется как fallback
        self.R_base = 2.5

    def predict(self, v, w, dt):
        theta = self.x[2]
        F = np.array([
            [1, 0, -v * math.sin(theta) * dt],
            [0, 1,  v * math.cos(theta) * dt],
            [0, 0,  1]
        ])
        self.x[0] += v * math.cos(theta) * dt
        self.x[1] += v * math.sin(theta) * dt
        self.x[2] += w * dt
        self.x[2] = (self.x[2] + math.pi) % (2 * math.pi) - math.pi
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z, beacon_pos, noise_std=None):
        """
        Обновление EKF по измерению дальности до маяка.
        noise_std — стандартное отклонение шума конкретного маяка.
        Если передан — используется вместо базового R.
        Это исправляет баг: раньше noise_std вычислялся, но в EKF не передавался,
        и матрица R всегда была фиксированной 2.5 независимо от качества сигнала.
        """
        bx, by = beacon_pos
        px, py = self.x[0], self.x[1]

        dist = math.hypot(px - bx, py - by)
        if dist < 1e-6:
            dist = 1e-6

        H = np.array([[(px - bx) / dist, (py - by) / dist, 0]])
        y_innov = z - dist

        # Адаптивная матрица R
        r_val = (noise_std ** 2) if noise_std is not None else self.R_base
        R = np.array([[r_val]])

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + (K * y_innov).flatten()
        self.x[2] = (self.x[2] + math.pi) % (2 * math.pi) - math.pi
        self.P = (np.eye(3) - K @ H) @ self.P