import random


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

    def run_simulation(self):
        self.initialize_verifiers()

        for _ in range(self.num_iterations):
            user = random.choice(self.users)
            map_claim = self.create_map_claim(user)
            self.auto_validate(map_claim)

            if map_claim.status == "verified":
                verifier_candidates = [u for u in self.users if u.role == "Verifier"]
                if verifier_candidates:
                    verifier = random.choice(verifier_candidates)
                    self.verify_map_claim(verifier, map_claim)

            if len([mc for mc in user.map_claims if mc.status == "rewarded"]) >= 5:
                user.role = "Verifier"

            if user.role == "Verifier" and random.random() < 0.1:  # 10% chance of being flagged
                flagger = random.choice(self.users)
                self.flag_verifier(flagger, user)

            # if random.random() < 0.05:  # 5% chance of a governance challenge
            #     governance = random.choice(self.users)
            #     flagged_verifier = random.choice([u for u in self.users if u.flags > 0])
            #     self.challenge_flags(governance, flagged_verifier)

        for user in self.users:
            print(f'User {user.id}: {user.tokens} tokens, '
                  f'{user.role} role, {user.flags} flags, '
                  f'{len([mc for mc in user.map_claims if mc.status == "verified"])} verified ,'
                  f'{len([mc for mc in user.map_claims if mc.status == "rejected"])} rejected ,'
                  f'{len([mc for mc in user.map_claims if mc.status == "invalid"])} invalid ,'
                  f'{len([mc for mc in user.map_claims if mc.status == "rewarded"])} rewarded ')

        # print total amount of map_claims for all user with status rewarded:
        print(f'Total amount of map_claims with status rewarded: {len([mc for mc in self.map_claims if mc.status == "rewarded"])}')

    def initialize_verifiers(self):
        num_verifiers = int(len(self.users) * self.num_verifier_ratio)

        # set random sample of users to role verifiers by using num_verifiers as sample size
        for user in random.sample(self.users, num_verifiers):
            user.role = "Verifier"


if __name__ == '__main__':
    num_users = 100000
    num_iterations = 1000
    num_verifier_ratio = 0.1

    simulation = Simulation(num_users, num_iterations, num_verifier_ratio)
    simulation.run_simulation()
