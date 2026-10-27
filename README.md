# Flappy Bird DQN Agent

A Deep Q-Network (DQN) agent that learns to play Flappy Bird from scratch using PyTorch and Gymnasium. It uses experience replay and a target network to stabilize training.

## Features

- Deep Q-Network with a 2-layer fully connected network (256 hidden units)
- Experience replay buffer with FIFO eviction and random mini-batch sampling
- Separate target network, periodically synced with the policy network
- Epsilon-greedy exploration with decay
- Hyperparameters configured in parameters.yaml
- Automatic saving of the best-performing model
- Episode logging to runs/<param_set>.log
- Automatic device selection: Apple MPS, CUDA, or CPU

## Project Structure

    FlappyBird_RL/
    ├── agent.py              # Training loop and optimization
    ├── dqn.py                # Q-network definition
    ├── experience_replay.py  # Replay buffer
    ├── parameters.yaml       # Hyperparameter sets
    └── runs/                 # Saved models and logs (auto-created)

## Installation

    pip install torch gymnasium flappy-bird-gymnasium pyyaml

Requires Python 3.8+.

## Usage

Train the agent:

    python agent.py flappybirdv0 --train

Training prints episode rewards and saves the best model to runs/flappybirdv0.pt.

Run the trained agent (renders the game window):

    python agent.py flappybirdv0

## Hyperparameters

| Parameter | Description | Value |
|---|---|---|
| epsilon_init | Initial exploration rate | 1 |
| epsilon_min | Minimum exploration rate | 0.05 |
| epsilon_decay | Epsilon decay per episode | 0.9995 |
| replay_memory_size | Replay buffer capacity | 100000 |
| mini_batch_size | Transitions per training step | 32 |
| network_sync_rate | Steps between target network syncs | 10 |
| alpha | Learning rate | 0.001 |
| gamma | Discount factor | 0.99 |
| reward_threshold | Episode reward at which an episode ends | 1000 |

## How It Works

1. The agent observes the current game state.
2. It picks an action with epsilon-greedy exploration.
3. The transition is stored in the replay buffer.
4. A random mini-batch is sampled, and the policy network is trained toward the Bellman target computed by the target network.
5. The target network is periodically synced with the policy network.

## Results

- Best episode reward: ~17 (reached at episode 21,040)
- Trained for 21,000+ episodes
- Training progress is logged in runs/flappybirdv0.log

## Acknowledgments

- [flappy-bird-gymnasium](https://github.com/markub3327/flappy-bird-gymnasium) for the environment
