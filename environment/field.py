import pygame

class RectangularField:
    def __init__(self, x, y, w, h):
        self.rect = pygame.Rect(x, y, w, h)

    def contains(self, x, y):
        return self.rect.collidepoint(x, y)

    def draw(self, screen):
        pygame.draw.rect(screen, (180, 180, 180), self.rect, 2)