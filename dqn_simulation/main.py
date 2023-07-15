import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque


class User:
    def __init__(self, id):
        self.id = id
        self.tokens = 0
        self.map_claims = []
        self.role = "Contributor"
        self.flags = 0
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_network()

    def build_model(self):
        model = tf.keras.Sequential()
        model.add(keras.layers.Dense(24, activation="relu", input_shape=(3,)))
        model.add(keras.layers.Dense(24, activation="linear"))
        model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        return model

    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())


class MapClaim:
    def __init__(self, user_id, status="unverified"):
        self.user_id = user_id
        self.status = status


class Simulation:
    def __init__(self, num_users, num_iterations, num_verifier_ratio, epsilon=0.1, gamma=0.95, batch_size=64):
        self.users = [User(i) for i in range(num_users)]
        self.map_claims = []
        self.num_iterations = num_iterations
        self.num_verifier_ratio = num_verifier_ratio
        self.epsilon = epsilon
        self.gamma = gamma
        self.batch_size = batch_size

    def create_map_claim(self, user):
        map_claim = MapClaim(user.id)
        user.map_claims.append(map_claim)
        self.map_claims.append(map_claim)
        return map_claim

    def auto_validate(self, map_claim):
        if random.random() < 0.9:  # 90% chance of being valid
            map_claim.status = "verified"
        else:
            map_claim.status = "invalid"

    def verify_map_claim(self, verifier, map_claim):
        if map_claim.status == "verified":
            if random.random() < 0.95:  # 95% chance of being rewarded
                map_claim.status = "rewarded"
                self.users[map_claim.user_id].tokens += 10
            else:
                map_claim.status = "rejected"

    def flag_verifier(self, user, verifier):
        verifier.flags += 1
        if verifier.flags >= 4:
            verifier.role = "Contributor"

    def challenge_flags(self, governance, flagged_verifier):
        if random.random() < 0.5:  # 50% chance of revoking flags
            flagged_verifier.flags = 0
            for user in self.users:
                if user.role == "Contributor":
                    user.tokens = 0

    def choose_action(self, user):
        if random.random() < self.epsilon:  # Exploration
            return random.randint(0, 2)
        else:  # Exploitation
            q_values = user.model.predict(np.identity(3)[np.newaxis, :])
            return np.argmax(q_values[0])

    def update_model(self, user, action, reward):
        q_values = user.model.predict(np.identity(3)[np.newaxis, :])[0]
        q_values_target = user.target_model.predict(np.identity(3)[np.newaxis, :])[0]

        target = q_values[action]
        if reward is not None:
            target = reward + self.gamma * np.max(q_values_target)

        q_values[action] = target
        user.model.fit(np.identity(3)[np.newaxis, :], q_values[np.newaxis, :], epochs=1, verbose=0)

    def run_simulation(self):
        self.initialize_verifiers()

        for _ in range(self.num_iterations):
            user = random.choice(self.users)
            action = self.choose_action(user)

            if action == 0:  # Create map claim
                map_claim = self.create_map_claim(user)
                self.auto_validate(map_claim)

                if map_claim.status == "verified":
                    verifier_candidates = [u for u in self.users if u.role == "Verifier"]
                    if verifier_candidates:
                        verifier = random.choice(verifier_candidates)
                        self.verify_map_claim(verifier, map_claim)

                if len([mc for mc in user.map_claims if mc.status == "rewarded"]) >= 5:
                    user.role = "Verifier"

                reward = 10 if map_claim.status == "rewarded" else -1
                self.update_model(user, action, reward)

            elif action == 1:  # Verify map claim
                if user.role == "Verifier":
                    map_claim_candidates = [mc for mc in self.map_claims if mc.status == "verified"]
                    if map_claim_candidates:
                        map_claim = random.choice(map_claim_candidates)
                        self.verify_map_claim(user, map_claim)

                        reward = 1
                        self.update_model(user, action, reward)

            elif action == 2:  # Flag verifier
                if user.role == "Verifier":
                    verifier_candidates = [u for u in self.users if u.role == "Verifier" and u.id != user.id]
                    if verifier_candidates:
                        verifier = random.choice(verifier_candidates)
                        self.flag_verifier(user, verifier)

                        reward = -1 if verifier.flags < 4 else 5
                        self.update_model(user, action, reward)

            if _ % self.batch_size == 0:
                user.update_target_network()

        for user in self.users:
            print(f'User {user.id}: {user.tokens} tokens, '
                  f'{user.role} role, {user.flags} flags')

        for user in self.users:
            print(f'User {user.id}: {user.tokens} tokens, '
                  f'{user.role} role, {user.flags} flags, '
                  f'{len([mc for mc in user.map_claims if mc.status == "verified"])} verified ,'
                  f'{len([mc for mc in user.map_claims if mc.status == "rejected"])} rejected ,'
                  f'{len([mc for mc in user.map_claims if mc.status == "invalid"])} invalid ,'
                  f'{len([mc for mc in user.map_claims if mc.status == "rewarded"])} rewarded ')

        # print total amount of map_claims for all user with status rewarded:
        print(f'Total amount of map_claims with status rewarded: {len([mc for mc in self.map_claims if mc.status == "rewarded"])}')
        print(f'Total amount of map_claims with status rejected: {len([mc for mc in self.map_claims if mc.status == "rejected"])}')
        print(f'Total amount of map_claims with status invalid: {len([mc for mc in self.map_claims if mc.status == "invalid"])}')
        print(f'Total amount of map_claims with status verified: {len([mc for mc in self.map_claims if mc.status == "verified"])}')

    def initialize_verifiers(self):
        num_verifiers = int(len(self.users) * self.num_verifier_ratio)

        # set random sample of users to role verifiers by using num_verifiers as sample size
        for user in random.sample(self.users, num_verifiers):
            user.role = "Verifier"


if __name__ == '__main__':
    num_users = 100
    num_iterations = 1000
    num_verifier_ratio = 0.1

    simulation = Simulation(num_users, num_iterations, num_verifier_ratio)
    simulation.run_simulation()
