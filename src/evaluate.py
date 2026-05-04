import pickle
import numpy as np
from collections import defaultdict

# same discretize function
def discretize(q):
    if q == 0:
        return 0
    elif q <= 3:
        return 1
    else:
        return 2

class TrafficEnv:
    def __init__(self, max_queue=10):
        self.max_queue = max_queue
        self.reset()

    def reset(self):
        self.queues = [0,0,0,0]
        self.phase = 0

    def step(self, action):
        if action == 1:
            self.phase = 1 - self.phase

        arrivals = np.random.randint(0,3,4)
        self.queues = [min(self.max_queue, q+a) for q,a in zip(self.queues,arrivals)]

        if self.phase == 0:
            self.queues[0] = max(0,self.queues[0]-3)
            self.queues[1] = max(0,self.queues[1]-3)
        else:
            self.queues[2] = max(0,self.queues[2]-3)
            self.queues[3] = max(0,self.queues[3]-3)

        reward = -sum(self.queues) - 2*max(self.queues)
        return reward

    def get_state(self):
        return tuple([discretize(q) for q in self.queues] + [self.phase])


# LOAD MODEL
with open("models/policy_final.pkl", "rb") as f:
    Q = defaultdict(lambda: [0,0], pickle.load(f))

# RL EVALUATION
env = TrafficEnv()
rl_reward = 0

for _ in range(2000):
    state = env.get_state()
    action = np.argmax(Q[state])
    rl_reward += env.step(action)

rl_avg = rl_reward / 2000


# BASELINE (fixed timer)
env = TrafficEnv()
baseline_reward = 0
phase = 0
counter = 0

for _ in range(2000):
    if counter >= 5:
        phase = 1 - phase
        counter = 0

    baseline_reward += env.step(0)
    counter += 1

baseline_avg = baseline_reward / 2000


print("\n--- FINAL RESULT ---")
print(f"Baseline Avg Reward: {round(baseline_avg,2)}")
print(f"RL Avg Reward: {round(rl_avg,2)}")