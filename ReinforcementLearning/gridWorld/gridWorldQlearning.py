import numpy as np
import matplotlib.pyplot as plt
import pygame
import sys


class GridWorld(object):

    def __init__(self, short_cut, grid):
        self.initial_grid = grid
        self.grid = self.initial_grid.copy()
        self.m = len(grid)
        self.n = len(grid)
        self.state_space = [i for i in range(self.m * self.n)]
        self.state_space.remove(80)
        self.state_space_plus = [i for i in range(self.m * self.n)]
        self.action_space = {'U': -self.m,
                             'D': self.m,
                             'L': -1,
                             'R': 1}
        self.possible_actions = ['U', 'D', 'L', 'R']
        self.add_short_cuts(short_cut)
        self.agent_pos = 0

    def add_short_cuts(self, short_cut):
        self.short_cut = short_cut
        i = 2
        for sq in short_cut:
            x = sq // self.m
            y = sq % self.n
            self.grid[x][y] = i
            i += 1
            x = short_cut[sq] // self.m
            y = short_cut[sq] % self.n
            self.grid[x][y] = i
            i += 1

    def is_term_state(self, state):
        return state in self.state_space_plus and state not in self.state_space

    def get_row_and_col(self):
        x = self.agent_pos // self.m
        y = self.agent_pos % self.n
        return x, y

    def set_state(self, state):
        x, y = self.get_row_and_col()
        self.grid[x][y] = 0
        self.agent_pos = state
        x, y = self.get_row_and_col()
        self.grid[x][y] = 1  # represents the agent

    def off_grid(self, new_state, old_state):
        if new_state not in self.state_space_plus:
            return True

        elif old_state % self.m == 0 and new_state % self.m == self.m - 1:
            return True

        elif old_state % self.m == self.m - 1 and new_state % self.m == 0:
            return True

        elif self.grid[new_state // self.m][new_state % self.n] == 9:
            return True

        else:
            return False

    def step(self, action):
        x, y = self.get_row_and_col()
        resulting_state = self.agent_pos + self.action_space[action]
        if resulting_state in self.short_cut.keys():
            resulting_state = self.short_cut[resulting_state]

        reward = -1 if not self.is_term_state(resulting_state) else 0
        if not self.off_grid(resulting_state, self.agent_pos):
            self.set_state(resulting_state)
            return resulting_state, reward, self.is_term_state(self.agent_pos), None
        else:
            return self.agent_pos, reward, self.is_term_state(self.agent_pos), None

    def reset(self):
        self.agent_pos = 0
        self.grid = self.initial_grid.copy()
        return self.agent_pos

    def render(self, screen):
        screen.fill((255, 255, 255))
        cell_size = 50
        for i in range(self.m):
            for j in range(self.n):
                x1 = j * cell_size
                y1 = i * cell_size
                rect = pygame.Rect(x1, y1, cell_size, cell_size)
                if self.grid[i][j] == 0:
                    pygame.draw.rect(screen, (255, 255, 255), rect)
                elif self.grid[i][j] == 1:
                    pygame.draw.rect(screen, (0, 0, 255), rect)
                elif self.grid[i][j] == 2:
                    pygame.draw.rect(screen, (0, 255, 0), rect)
                elif self.grid[i][j] == 3:
                    pygame.draw.rect(screen, (144, 238, 144), rect)
                elif self.grid[i][j] == 4:
                    pygame.draw.rect(screen, (255, 0, 0), rect)
                elif self.grid[i][j] == 5:
                    pygame.draw.rect(screen, (255, 192, 203), rect)
                elif self.grid[i][j] == 9:
                    pygame.draw.rect(screen, (0, 0, 0), rect)
                pygame.draw.rect(screen, (0, 0, 0), rect, 1)

    def action_space_sample(self):
        return np.random.choice(self.possible_actions)


def max_action(Q, state, actions):
    values = np.array([Q[state, a] for a in actions])
    return actions[np.argmax(values)]


if __name__ == '__main__':
    short_cut = {18: 54, 63: 14}
    grid = np.array([
            [0, 0, 9, 0, 0, 0, 0, 0, 0],
            [0, 0, 9, 0, 0, 0, 0, 0, 0],
            [0, 0, 9, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [9, 9, 9, 9, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 9, 9, 9],
            [0, 0, 0, 0, 0, 0, 0, 0, 0]
        ])
    env = GridWorld(short_cut, grid)

    LR = 0.1
    GAMMA = 1.0
    EPS = 1.0

    Q = {}

    for state in env.state_space_plus:
        for action in env.possible_actions:
            Q[state, action] = 0  # encourage exploration

    num_games = 50000
    total_rewards = np.zeros(num_games)

    pygame.init()
    screen = pygame.display.set_mode((450, 450))
    pygame.display.set_caption('GridWorld')

    for i in range(num_games):
        if i % 5000 == 0:
            print('Starting game ', i)
        done = False
        ep_rewards = 0
        observation = env.reset()

        while not done:
            """ FOR VISUALIZATION 
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            env.render(screen)
            pygame.display.flip()
            """
            rand = np.random.random()
            action = max_action(Q, observation, env.possible_actions) if rand < (1 - EPS) else env.action_space_sample()
            observation_, reward, done, info = env.step(action)
            ep_rewards += reward
            action_ = max_action(Q, observation_, env.possible_actions)
            Q[observation, action] = Q[observation, action] + LR * (
                    reward + GAMMA * Q[observation_, action_] - Q[observation, action])

            observation = observation_

            if EPS - 2 / num_games > 0:
                EPS -= 2 / num_games
            else:
                EPS = 0

        total_rewards[i] = ep_rewards

    plt.plot(total_rewards)
    plt.show()

