import numpy as np
import pygame
import time
import random

display_width = 800
display_height = 800

white = (255, 255, 255)
black = (0, 0, 0)
red = (255, 0, 0)
green = (0, 255, 0)

gameDisplay = pygame.display.set_mode((display_width, display_height))

ship_size = 50
astroid_size = 400

def ship(x, y):
    pygame.draw.rect(gameDisplay, green, [x, y, ship_size, ship_size])

def astroid(x, y):
    pygame.draw.rect(gameDisplay, red, [x, y, astroid_size, astroid_size])

def text_object(text, font):
    textSurface = font.render(text, True, white)
    return textSurface, textSurface.get_rect()

def crash():
    print('Crash!')

def reset_game():
    return Game().reset()

class Game:
    def __init__(self):
        self.score = 0
        self.x = display_width * 0.45
        self.y = display_height * 0.8
        self.x_change = 0
        self.y_change = 0
        self.movement = 10
        self.astroid_startX = random.randrange(0, display_width)
        self.astroid_startY = -500
        self.speed = 10
        self.astroid_width = astroid_size
        self.astroid_height = astroid_size
        self.clock = pygame.time.Clock()

    def reset(self):
        self.__init__()
        self.score = 0
        return self.get_state()

    def is_collision(self, x=None, y=None):
        if x is None:
            x = self.x
        if y is None:
            y = self.y

        if x > display_width - ship_size or x < 0 or y > display_height - ship_size or y < 0:
            return True

        if (self.astroid_startX < x < self.astroid_startX + self.astroid_width or
            self.astroid_startX < x + ship_size < self.astroid_startX + self.astroid_width):
            if (self.astroid_startY < y < self.astroid_startY + self.astroid_height or
                self.astroid_startY < y + ship_size < self.astroid_startY + self.astroid_height):
                return True

        return False

    def get_state(self):
        state = [
            self.x / display_width,
            self.y / display_height,
            self.astroid_startX / display_width,
            self.astroid_startY / display_height,
            (self.astroid_startX - self.x) / display_width,
            (self.astroid_startY - self.y) / display_height
        ]
        return np.array(state, dtype=np.float32)

    def step(self, action):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self.x_change = 0

        if action == 0:  # LEFT
            self.x_change = -self.movement
        elif action == 1:  # RIGHT
            self.x_change = self.movement
        elif action == 2:  # NOTHING
            self.x_change = 0

        self.x += self.x_change

        gameDisplay.fill(black)

        astroid(self.astroid_startX, self.astroid_startY)
        self.astroid_startY += self.speed

        ship(self.x, self.y)

        if self.is_collision():
            return self.get_state(), -100, True, self.score  # Large negative reward for collision

        if self.astroid_startY > display_height:
            self.astroid_startY = 0 - self.astroid_height
            self.astroid_startX = random.randrange(0, display_width)
            self.score += 1
            return self.get_state(), 10, False, self.score  # Positive reward for passing an asteroid

        reward = 0.1  # Small reward for step survived

        # Add a negative reward for proximity to walls
        wall_proximity_threshold = 50
        if self.x < wall_proximity_threshold or self.x > display_width - wall_proximity_threshold:
            reward -= 1  # Penalize for being close to the vertical walls

        pygame.display.update()
        self.clock.tick(30)

        return self.get_state(), reward, False, self.score

