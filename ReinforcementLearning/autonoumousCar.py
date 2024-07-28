import gym
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import os

# Define helper function to create animation
def create_animation(frames, title):
    fig, ax = plt.subplots()
    ax.axis('off')
    img = ax.imshow(frames[0])

    def update(frame):
        img.set_data(frame)
        return img,

    ani = FuncAnimation(fig, update, frames=frames, blit=True, interval=50)
    plt.title(title)
    plt.show()

environment_name = "CarRacing-v2"

env = gym.make(environment_name, render_mode="rgb_array")
frames = []
episodes = 5
for episode in range(1, episodes + 1):
    state = env.reset()
    done = False
    score = 0

    while not done:
        frame = env.render()
        frames.append(frame)
        action = env.action_space.sample()
        n_state, reward, done, trunc, info = env.step(action)
        score += reward
    print(f'Episode: {episode} Score: {score}')
env.close()

env = DummyVecEnv([lambda: gym.make(environment_name, render_mode="rgb_array")])

log_path = os.path.join('Training', 'Logs')
model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=log_path)
model.learn(total_timesteps=100000)  # Increased timesteps

ppo_path = os.path.join('Training', 'Saved_Models', 'PPO_Driving_model')
os.makedirs(os.path.dirname(ppo_path), exist_ok=True)
model.save(ppo_path)
model = PPO.load(ppo_path, env=env)

frames = []
obs = env.reset()
done = False
while not done:
    frame = env.render()
    frames.append(frame)
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)

env.close()

create_animation(frames, title="Trained PPO Agent")
