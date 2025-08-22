import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")  # görsel
obs, info = env.reset(seed=42)

for _ in range(1000):
    action = env.action_space.sample()  # rastgele action
    obs, reward, done, truncated, info = env.step(action)
    if done or truncated:
        obs, info = env.reset()

env.close()
