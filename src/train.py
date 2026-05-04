import numpy as np
import random
import matplotlib.pyplot as plt
import pickle
import yaml
import argparse
import csv
import os
from collections import defaultdict


# -------------------------------
# LOAD CONFIG
# -------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True)
args = parser.parse_args()

with open(args.config, "r") as f:
    config = yaml.safe_load(f)

print("Loaded Config:", config)


# -------------------------------
# DISCRETIZATION FUNCTION
# -------------------------------
def discretize(q):
    if q == 0:
        return 0
    elif q <= 3:
        return 1
    else:
        return 2


# -------------------------------
# ENVIRONMENT
# -------------------------------
class TrafficEnv:
    def __init__(self, max_queue):
        self.max_queue = max_queue
        self.reset()

    def reset(self):
        self.queues = [0, 0, 0, 0]
        self.phase = 0
        return self._get_state()

    def step(self, action):
        if action == 1:
            self.phase = 1 - self.phase

        arrivals = np.random.randint(0, 3, size=4)
        self.queues = [min(self.max_queue, q + a) for q, a in zip(self.queues, arrivals)]

        if self.phase == 0:
            self.queues[0] = max(0, self.queues[0] - 3)
            self.queues[1] = max(0, self.queues[1] - 3)
        else:
            self.queues[2] = max(0, self.queues[2] - 3)
            self.queues[3] = max(0, self.queues[3] - 3)

        reward = -sum(self.queues) - 2 * max(self.queues)

        return self._get_state(), reward

    def _get_state(self):
        d_queues = [discretize(q) for q in self.queues]
        return tuple(d_queues + [self.phase])


# -------------------------------
# MOVING AVERAGE
# -------------------------------
def moving_average(data, window=50):
    return np.convolve(data, np.ones(window)/window, mode='valid')


# -------------------------------
# TRAIN RL
# -------------------------------
def train():
    env = TrafficEnv(config["max_queue"])
    Q = defaultdict(lambda: [0, 0])

    alpha = config["alpha"]
    gamma = config["gamma"]
    epsilon = config["epsilon"]
    epsilon_decay = config["epsilon_decay"]

    episodes = config["episodes"]
    steps = config["steps_per_episode"]

    rewards = []

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0

        for _ in range(steps):

            # ε-greedy
            if random.random() < epsilon:
                action = random.choice([0, 1])
            else:
                action = np.argmax(Q[state])

            next_state, reward = env.step(action)

            best_next = max(Q[next_state])
            Q[state][action] += alpha * (reward + gamma * best_next - Q[state][action])

            state = next_state
            total_reward += reward

        rewards.append(total_reward)

        # decay epsilon
        epsilon = max(0.01, epsilon * epsilon_decay)

        if (ep + 1) % 50 == 0:
            print(f"Episode {ep+1}, Reward: {total_reward}, Epsilon: {round(epsilon, 3)}")

    return Q, rewards


# -------------------------------
# SAVE RESULTS (CSV)
# -------------------------------
def save_results(rewards):
    avg_reward = sum(rewards[-50:]) / 50

    os.makedirs("experiments", exist_ok=True)
    file_path = "experiments/results.csv"

    file_exists = os.path.isfile(file_path)

    with open(file_path, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["run_id", "episodes", "avg_reward", "alpha", "gamma", "epsilon"])

        writer.writerow([
            config["id"],
            config["episodes"],
            round(avg_reward, 2),
            config["alpha"],
            config["gamma"],
            config["epsilon"]
        ])

    print("Results saved to experiments/results.csv")


# -------------------------------
# MAIN
# -------------------------------
Q, rewards = train()

# Plot
smoothed = moving_average(rewards)
plt.plot(smoothed)
plt.xlabel("Episodes")
plt.ylabel("Smoothed Reward")
plt.title("RL Learning Curve")
plt.show()

# Save model
os.makedirs("models", exist_ok=True)
with open("models/policy_final.pkl", "wb") as f:
    pickle.dump(dict(Q), f)

print("Model saved to models/policy_final.pkl")

# Save experiment results
save_results(rewards)