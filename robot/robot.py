import math
import random


class Robot:
    def __init__(self, x, y, theta=0.0):
        self.x = x
        self.y = y
        self.theta = theta
        self.width = 20

        # Шум энкодеров одометрии
        self.noise_v     = 1.2    # случайный шум скорости (px/s)
        self.noise_w     = 0.04   # случайный шум угловой скорости (rad/s)
        self.drift_rate  = 0.004  # систематический дрейф: 0.4% от скорости
        self._angle_bias = random.gauss(0, 0.008)  # постоянная ошибка гироскопа

    def move(self, v, w, dt):
        self.theta += w * dt
        self.theta = (self.theta + math.pi) % (2 * math.pi) - math.pi
        self.x += v * math.cos(self.theta) * dt
        self.y += v * math.sin(self.theta) * dt

    def get_odometry(self, v, w, dt):
        """
        Зашумлённые показания одометрии — то, что видит EKF и DR-трекер.
        Случайный шум + систематический дрейф + постоянный угловой bias.
        Без этих ошибок EKF был бы бесполезен.
        """
        noisy_v = v + random.gauss(0, self.noise_v)
        noisy_w = w + random.gauss(0, self.noise_w)
        noisy_v += v * self.drift_rate * random.uniform(0.5, 1.5)
        noisy_w += self._angle_bias
        return noisy_v, noisy_w