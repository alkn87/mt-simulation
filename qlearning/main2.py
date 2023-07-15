import numpy as np
import tensorflow as tf
from tensorflow import keras

# Constants
NUM_STATES = 2
NUM_ACTIONS = 3
REWARD_MAP_CLAIM = 10
MIN_MAP_CLAIMS_TO_BECOME_VERIFIER = 5


# Environment
class MapClaimEnv:
    def __init__(self):
        self.state = 0
        self.num_verified_map_claims = 0

    def step(self, action):
        reward = 0
        done = False

        if action == 0:  # Create a new map claim
            self.state = 1
        elif action == 1:  # Flag another user with role 'verifier'
            if self.state == 1:
                self.state = 0
                reward = -1
            else:
                reward = -1
        elif action == 2:  # Verify a map claim from another user
            if self.state == 1:
                self.num_verified_map_claims += 1
                self.state = 0
                reward = REWARD_MAP_CLAIM

        if self.num_verified_map_claims >= MIN_MAP_CLAIMS_TO_BECOME_VERIFIER:
            done = True

        return self.state, reward, done


# Q-Network
class QNetwork(tf.keras.Model):
    def __init__(self, num_states, num_actions):
        super(QNetwork, self).__init__()
        self.dense1 = keras.layers.Dense(32, activation='relu')
        self.dense2 = keras.layers.Dense(32, activation='relu')
        self.dense3 = keras.layers.Dense(num_actions)

    def call(self, inputs):
        x = self.dense1(tf.reshape(inputs, [-1, NUM_STATES]))
        x = self.dense2(x)
        x = self.dense3(x)
        return x


# Q-Learning Agent
class QLearningAgent:
    def __init__(self, num_states, num_actions, learning_rate=0.01, gamma=0.99, epsilon=1.0, epsilon_decay=0.995,
                 epsilon_min=0.01):
        self.q_network = QNetwork(num_states, num_actions)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def update_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, 3)
        else:
            state = np.array([state], dtype=np.float32)
            return np.argmax(self.q_network(tf.reshape(state, [-1, NUM_STATES])).numpy()[0])

    def train(self, state, action, next_state, reward, done):
        with tf.GradientTape() as tape:
            state = np.array([state], dtype=np.float32)
            next_state = np.array([next_state], dtype=np.float32)
            q_values = self.q_network(tf.reshape(state, [-1, NUM_STATES]))
            next_q_values = self.q_network(tf.reshape(next_state, [-1, NUM_STATES]))

            target_q_values = np.copy(q_values.numpy())
            target_q_values[0, action] = reward

            if not done:
                target_q_values[0, action] += self.gamma * np.max(next_q_values.numpy())

            loss = tf.keras.losses.MSE(q_values, target_q_values)

        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.q_network.trainable_variables))


def main():
    num_episodes = 1000
    max_steps_per_episode = 100

    env = MapClaimEnv()
    agent = QLearningAgent(NUM_STATES, NUM_ACTIONS)

    for episode in range(num_episodes):
        state = env.state
        total_reward = 0

        for step in range(max_steps_per_episode):
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.train(state, action, next_state, reward, done)
            total_reward += reward
            state = next_state

            if done:
                break

        agent.update_epsilon()
        print(f"Episode {episode + 1}: Total reward: {total_reward}")


if __name__ == "__main__":
    main()
