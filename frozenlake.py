import numpy as np
import gymnasium as gym

# Ortamı oluştur
env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="ansi")

n_states = env.observation_space.n
n_actions = env.action_space.n

# Q tablosu (state x action)
Q = np.zeros((n_states, n_actions))

# Hiperparametreler
alpha = 0.8        # öğrenme oranı
gamma = 0.95       # gelecek ödül katsayısı
epsilon = 1.0      # keşif oranı (başta yüksek)
epsilon_decay = 0.999
epsilon_min = 0.05
episodes = 5000

# Eğitim
for episode in range(episodes):
    state, _ = env.reset()
    done = False

    while not done:
        # Epsilon-greedy action seçimi
        if np.random.rand() < epsilon:
            action = env.action_space.sample()  # rastgele
        else:
            action = np.argmax(Q[state])

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Q-learning güncelleme
        Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])

        state = next_state

    # Epsilon azalt
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

print("Eğitim tamamlandı ✅")

# Test (eğitim sonrası)
state, _ = env.reset()
done = False
total_reward = 0

while not done:
    action = np.argmax(Q[state])  # artık greedy
    state, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward
    done = terminated or truncated

print("Testte elde edilen ödül:", total_reward)
