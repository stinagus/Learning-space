import numpy as np
import pygame

from ReinforcementLearning.starship.game import Game
from ReinforcementLearning.starship.model import DQNAgent

if __name__ == "__main__":
    pygame.init()
    game = Game()
    state_size = len(game.get_state())
    action_size = 3  # left, right, nothing
    agent = DQNAgent(state_size, action_size)
    episodes = 1000
    batch_size = 32

    for e in range(episodes):
        state = game.reset()
        state = np.reshape(state, [1, state_size])
        if agent.epsilon > agent.epsilon_min:
            print(agent.epsilon)
            agent.epsilon *= agent.epsilon_decay

        for time in range(500):
            action = agent.act(state)
            next_state, reward, done, score = game.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print(f"Episode: {e}/{episodes}, Score: {score}, Epsilon: {agent.epsilon:.2f}")
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        if e % 10 == 0:
            agent.update_target_model()

    pygame.quit()
