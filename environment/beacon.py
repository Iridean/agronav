import math
import random


class Beacon:
    def __init__(self, beacon_id, x, y, radius, beacon_map, fail_prob=0.0):
        self.id = beacon_id
        self.x = x
        self.y = y
        self.radius = radius
        self.fail_prob = fail_prob   # вероятность случайного отказа в кадр
        self.active = True
        self.neighbors = []
        # Счётчик для плавного «моргания» — не отказывает каждый кадр,
        # а держит отказ несколько кадров подряд (реалистичнее)
        self._fail_timer = 0

    def is_available(self, rx, ry):
        if not self.active:
            return False
        dist = math.hypot(rx - self.x, ry - self.y)
        if dist > self.radius:
            return False
        # Симуляция временного отказа: держим 0.5–1.5 сек (~30–90 кадров)
        if self._fail_timer > 0:
            self._fail_timer -= 1
            return False
        if random.random() < self.fail_prob:
            self._fail_timer = random.randint(30, 90)
            return False
        return True

    def get_noise_std(self, rx, ry, mesh_connected=False):
        """
        Динамическая модель шума измерения дальности.
        Учитывает три фактора:
          1. Расстояние: чем дальше — тем хуже сигнал (реальная UWB-модель)
          2. Mesh-связь: маяк в сети коллег точнее (дифференциальная коррекция)
          3. fail_prob: ненадёжный маяк добавляет шум даже когда работает
        """
        dist = math.hypot(rx - self.x, ry - self.y)
        # Базовый шум растёт линейно с расстоянием (реальный UWB: ~2 см/м)
        base = 1.0 + (dist / self.radius) * 2.5
        # Mesh-коррекция снижает шум в 3 раза (как дифференциальный GPS)
        if mesh_connected:
            base *= 0.35
        # Ненадёжный маяк добавляет шум пропорционально своей вероятности отказа
        base += self.fail_prob * 4.0
        return max(0.3, base)

    def measure_distance(self, rx, ry, noise_std=None):
        true_dist = math.hypot(rx - self.x, ry - self.y)
        std = noise_std if noise_std is not None else 1.5
        return true_dist + random.gauss(0, std)