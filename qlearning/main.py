import numpy as np
import random
from collections import deque
# import tensorflow as tf
# from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam


# Define the Deep Q-learning agent class
class DeepQLearningAgent:
    def __init__(self, num_actions, num_users):
        self.num_actions = num_actions
        self.num_users = num_users
        self.memory = deque(maxlen=2000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.01
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.num_users, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.num_actions, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.choice(range(self.num_actions))
        else:
            q_values = self.model.predict(state)
            return np.argmax(q_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state in minibatch:
            target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# Constants
NUM_ACTIONS = 4
NUM_USERS = 10
NUM_ROUNDS = 10
BATCH_SIZE = 32

# Initialize agent and game state
agent = DeepQLearningAgent(NUM_ACTIONS, NUM_USERS)
game_state = np.zeros(NUM_USERS)
flags = np.zeros(NUM_USERS)
verified_claims = np.zeros(NUM_USERS)

# Main loop
for _ in range(NUM_ROUNDS):
    for user in range(NUM_USERS):
        state = np.reshape(game_state, [1, NUM_USERS])

        # Agent chooses an action
        action = agent.choose_action(state)

        # Perform action and update game state
        reward = 0
        new_user = (user + 1) % NUM_USERS

        if action == 0:  # Create a map claim
            game_state[user] += 1
        elif action == 1:  # Flag another user
            flags[new_user] += 1
            if flags[new_user] >= 5:
                reward = -1
        elif action == 2:  # Verify a map claim
            if game_state[new_user] > 0:
                reward = 10
                verified_claims[new_user] += 1
                game_state[new_user] -= 1
        else:  # No operation
            pass

        # Remember the action and experience replay
        next_state = np.reshape(game_state, [1, NUM_USERS])
        agent.remember(state, action, reward, next_state)
        if len(agent.memory) >= BATCH_SIZE:  # Add this condition
            agent.replay(BATCH_SIZE)

print("Verified claims:")
