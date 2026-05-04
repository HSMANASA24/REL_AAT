# 🚦 Adaptive Traffic Signal Control using Reinforcement Learning

## 📌 Overview

This project implements a **Q-learning based traffic signal controller** that dynamically adjusts signal phases at a four-way intersection to reduce congestion and vehicle waiting time.

The system is compared against a **fixed-timer baseline**, demonstrating the effectiveness of reinforcement learning in real-time traffic optimization.

---

## 🎯 Problem Statement

Design and implement a reinforcement learning-based traffic signal controller that dynamically selects signal phases to minimize vehicle waiting time and congestion.

---

## 🌍 SDG Alignment

This project supports **SDG 11 – Sustainable Cities and Communities** by:

* Reducing traffic congestion
* Lowering fuel consumption
* Minimizing urban air pollution

---

## 🧠 Methodology

### 🔹 State

* Queue length in each direction (N, S, E, W)
* Current signal phase

### 🔹 Action

* 0 → Keep current signal
* 1 → Switch signal

### 🔹 Reward

* Negative of total congestion (queue length)
* Additional penalty for worst queue

### 🔹 Algorithm

* **Q-learning (tabular RL)**
* ε-greedy exploration strategy
* Epsilon decay for convergence

---

## ⚙️ MLOps Implementation

### ✔ Reproducibility

* Hyperparameters stored in YAML config (`configs/qlearning.yaml`)

### ✔ Experiment Tracking

* Results logged in `experiments/results.csv`

### ✔ Version Control

* Git used for tracking experiment versions

---

## 📁 Project Structure

```
REL_AAT/
 ├── src/
 │    ├── train.py
 │    └── evaluate.py
 ├── configs/
 │    └── qlearning.yaml
 ├── experiments/
 │    └── results.csv
 ├── models/
 │    └── policy_final.pkl
 └── README.md
```

---

## ▶️ How to Run

### 1. Train the model

```
python src/train.py --config configs/qlearning.yaml
```

### 2. Evaluate performance

```
python src/evaluate.py
```

---

## 📊 Results

| Method      | Avg Reward |
| ----------- | ---------- |
| Fixed Timer | -39.92     |
| RL Policy   | -9.3       |

👉 The RL model significantly reduces congestion compared to the baseline.

---

## 📈 Observations

* RL adapts dynamically to traffic conditions
* Fixed timer fails under varying traffic loads
* Training shows improving reward trend with minor fluctuations

---

## 🧠 Key Insight

The RL agent learns optimal switching strategies by interacting with the environment, leading to better traffic flow management compared to static approaches.

---

## 🎤 Conclusion

Reinforcement learning provides an effective solution for adaptive traffic signal control, significantly improving traffic efficiency and supporting sustainable urban development.

---

## 🚀 Future Work

* Multi-intersection coordination
* Deep Q-learning (DQN)
* Real-world deployment with sensors

---

