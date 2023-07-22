import random
import matplotlib.pyplot as plt


class User:
    def __init__(self, id):
        self.id = id
        self.tokens = 0
        self.map_claims = []
        self.role = "Contributor"
        self.flags = 0


class MapClaim:
    def __init__(self, user_id, status="unverified"):
        self.user_id = user_id
        self.status = status


class Simulation:
    def __init__(self, num_users, num_iterations, num_verifier_ratio):
        self.users = [User(i) for i in range(num_users)]
        self.map_claims = []
        self.num_iterations = num_iterations
        self.num_verifier_ratio = num_verifier_ratio
        self.total_rewards_over_time = []
        self.num_contributors_over_time = []
        self.num_verifiers_over_time = []
        self.num_flagged_over_time = []
        self.num_demotions_over_time = []
        self.num_promotions_over_time = []
        self.num_demotions = 0
        self.num_promotions = 0

    def create_map_claim(self, user):
        map_claim = MapClaim(user.id)
        user.map_claims.append(map_claim)
        self.map_claims.append(map_claim)
        return map_claim

    def promote_user(self, user):
        if user.tokens > 50:
            user.role = "Verifier"
            self.num_promotions += 1

    def auto_validate(self, map_claim):
        if random.random() < 0.9:  # 90% chance of being valid
            map_claim.status = "verified"
        else:
            map_claim.status = "rejected"

    def verify_map_claim(self, verifier, map_claim):
        if map_claim.status == "verified" and verifier.id != map_claim.user_id:
            if random.random() < 0.95:  # 95% chance of being rewarded
                map_claim.status = "rewarded"
                self.users[map_claim.user_id].tokens += 10
                if self.users[map_claim.user_id].tokens > 50:
                    self.promote_user(self.users[map_claim.user_id])
            else:
                map_claim.status = "rejected"

    def flag_verifier(self, verifier):
        verifier.flags += 1
        if verifier.flags >= 4:
            verifier.role = "Contributor"
            verifier.tokens = 0
            self.num_demotions += 1

    def challenge_flags(self, governance, flagged_verifier):
        if random.random() < 0.5:  # 50% chance of revoking flags
            flagged_verifier.flags = 0
            for user in self.users:
                if user.role == "Contributor":
                    user.tokens = 0

    def run_simulation(self):
        self.initialize_verifiers()

        for _ in range(self.num_iterations):
            # Select a random user and action
            user = random.choice([u for u in self.users if u.role == "Contributor"])
            user_action = random.choices(
                ["create", "do_nothing", "flag"],
                weights=[0.7, 0.20, 0.10],
                k=1)[0]

            if user_action == "create":
                map_claim = self.create_map_claim(user)
                self.auto_validate(map_claim)
            elif user_action == "flag" and [u for u in self.users if u.role == "Verifier"] and user.tokens > 10:
                verifier = random.choice([u for u in self.users if u.role == "Verifier"])
                self.flag_verifier(verifier)

            # Select a random verifier and action
            verifier_candidates = [u for u in self.users if u.role == "Verifier"]

            if verifier_candidates:
                verifier = random.choice([u for u in self.users if u.role == "Verifier"])
                verifier_action = random.choices(
                    ["check", "do_nothing", "create"],
                    weights=[0.6, 0.2, 0.2],
                    k=1)[0]

                if verifier_action == "check" and self.map_claims:
                    map_claim = random.choice(self.map_claims)
                    self.verify_map_claim(verifier, map_claim)

                if verifier_action == "create":
                    map_claim = self.create_map_claim(verifier)
                    self.auto_validate(map_claim)

            self.total_rewards_over_time.append(len([mc for mc in self.map_claims if mc.status == "rewarded"]))
            self.num_contributors_over_time.append(len([u for u in self.users if u.role == "Contributor"]))
            self.num_verifiers_over_time.append(len([u for u in self.users if u.role == "Verifier"]))
            self.num_flagged_over_time.append(len([u for u in self.users if u.flags > 0]))
            self.num_demotions_over_time.append(self.num_demotions)
            self.num_promotions_over_time.append(self.num_promotions)

            # if random.random() < 0.05:  # 5% chance of a governance challenge
            #     governance = random.choice(self.users)
            #     flagged_verifier = random.choice([u for u in self.users if u.flags > 0])
            #     self.challenge_flags(governance, flagged_verifier)

        for user in self.users:
            print(f'User {user.id}: {user.tokens} tokens, '
                  f'{user.role} role, {user.flags} flags, '
                  f'{len([mc for mc in user.map_claims if mc.status == "verified"])} verified ,'
                  f'{len([mc for mc in user.map_claims if mc.status == "rejected"])} rejected ,'
                  f'{len([mc for mc in user.map_claims if mc.status == "rewarded"])} rewarded ')

        # print total amount of map_claims for all user with status rewarded:
        print(
            f'Total amount of map_claims with status rewarded: {len([mc for mc in self.map_claims if mc.status == "rewarded"])}')

    def initialize_verifiers(self):
        num_verifiers = int(len(self.users) * self.num_verifier_ratio)

        # set random sample of users to role verifiers by using num_verifiers as sample size
        for user in random.sample(self.users, num_verifiers):
            user.role = "Verifier"


if __name__ == '__main__':
    num_users = 2000
    num_iterations = 10000
    num_verifier_ratio = 0.1

    simulation = Simulation(num_users, num_iterations, num_verifier_ratio)
    simulation.run_simulation()

    # Collect token data
    token_data = [user.tokens for user in simulation.users]

    # Create histogram
    plt.hist(token_data, bins=range(0, max(token_data) + 10, 10))
    plt.title("Token Distribution Among Users \n" + f"Users: {num_users} Iterations: {num_iterations} Verifier Ratio: {num_verifier_ratio}")
    plt.xlabel("Token Amount")
    plt.ylabel("Number of Users")
    plt.text(0.5 * num_iterations, 0.5 * max(token_data) + 10,
             f"Users: {num_users}\nIterations: "
             f"{num_iterations}\nVerifier Ratio: "
             f"{num_verifier_ratio}", horizontalalignment='center')
    plt.show()

    # Plot number of flagged users and demotions over time
    plt.plot(simulation.num_flagged_over_time, label="Flagged Verifiers")
    plt.plot(simulation.num_demotions_over_time, label="Demotions")
    plt.plot(simulation.num_promotions_over_time, label="Promotions")
    plt.title("Number of Flagged Verifiers, Demotions, Promotions Over Time")
    plt.xlabel("Iterations")
    plt.ylabel("Count")
    plt.text(0.5 * num_iterations, 0.5 * max(simulation.num_flagged_over_time +
                                             simulation.num_demotions_over_time +
                                             simulation.num_promotions_over_time), f"Users: {num_users}\nIterations: "
                                                                                   f"{num_iterations}\nVerifier Ratio: "
                                                                                   f"{num_verifier_ratio}", horizontalalignment='center')
    plt.legend()  # Display a legend
    plt.show()
