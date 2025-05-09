# Deep Reinforcement Learning for Inventory Management

This project investigates the application of Deep Reinforcement Learning (DRL) to multi-echelon inventory management problems characterized by stochastic demand and lead times. It evaluates both standard DRL algorithms and novel heuristic-shaped variants (HuRL) to optimize replenishment decisions.

## 🔍 Overview
We design a discrete-time simulation environment that models inventory dynamics across varying prices and lead times. The environment supports the training and evaluation of:

- **Standard DRL agents**: DQN, A2C, PPO, SAC
- **Heuristic-guided agents (HuRL)**: HuDQN, HuA2C, HuPPO, HuSAC
- **Classical heuristic policies**: Basestock, Capped Basestock, Constant Order, Myopic

## 🚀 Features
- Simulation of inventory systems with configurable price and lead time scenarios
- Full implementation of DRL and HuRL algorithms with modular training pipelines
- Integrated grid search for hyperparameter tuning
- Comparative evaluation against multiple heuristic baselines
- Clean and reproducible experiment design with detailed result logging

## 📁 Folder Structure
```
├── agents/               # Implementations of DQN, A2C, PPO, SAC
├── heuristics/           # Classical policies and heuristic functions
├── hurl/                 # HuRL algorithm extensions
├── env/                  # Inventory simulation environment
├── results/              # Output data and plots
├── configs/              # Hyperparameter tuning grids
├── train_rl.py           # Training of rl agents
├── train_hurl.py         # Training of hurl agents
├── utils.py              # Utility functions
├── train_rl.py           # Hyperparameters and env parameters grids
└── README.md
```

HuRL consistently improves over its DRL counterpart, with notable gains in stability and final policy value.

## 📜 License
MIT License

---
© 2025 National University of Singapore. All rights reserved.
