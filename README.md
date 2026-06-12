# Core-RL Framework

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

Core-RL is a modular, plug-and-play Reinforcement Learning (RL) framework designed to act as the "brain" for diverse applications, ranging from autonomous driving to robotics and quantitative finance. 

Unlike traditional static RL setups, Core-RL is built with **Human-in-the-Loop (Active Learning)** and **Continual Learning** capabilities. It is designed to ask for human guidance when uncertain and learn new environments without forgetting previously acquired skills.

## Project Purpose & Key Features

The primary goal of this framework is to provide a generic, environment-agnostic RL core. You define the environment and the reward; Core-RL handles the complex mathematics, memory management, and neural network updates.

* **Generic Architecture:** Abstracted agent interfaces (SAC, PPO, DQN) that work seamlessly with any custom environment (visual, numerical, or multimodal).
* **Active Learning (Human-in-the-Loop):** Monitors policy entropy (uncertainty). If the agent faces an unknown edge case, it pauses and routes the data to a web UI for human annotation, converting manual feedback into direct rewards.
* **Continual Learning:** Implements Elastic Weight Consolidation (EWC) to prevent catastrophic forgetting. The agent can learn new tasks (e.g., new traffic signs, new market conditions) while preserving critical weights from past experiences.
* **Multimodal Replay Buffer:** A robust memory system capable of handling nested dictionaries, images, and standard vector states simultaneously.

## Technologies Used

* **Deep Learning:** [PyTorch](https://pytorch.org/)
* **RL Environments:** [Gymnasium](https://gymnasium.farama.org/)
* **Numerical Computation:** NumPy
* **Feedback API / Backend:** Flask (for Active Learning UI)
* **Testing:** Pytest

## Folder Structure

```text
core-rl-framework/
├── core_rl/
│   ├── __init__.py
│   ├── agents/              # Abstract Base Agent and specific implementations (SAC, PPO)
│   ├── buffer/              # Multimodal Replay Buffer for experience replay
│   ├── continual/           # Continual learning algorithms (e.g., EWC)
│   └── active_learning/     # Entropy calculation and human-review queue management
├── feedback_engine/         # Web backend for human-in-the-loop annotations
├── tests/                   # Unit tests for core components
├── .github/workflows/       # CI/CD pipelines
├── README.md                
├── requirements.txt         # Project dependencies
└── setup.py                 # Package configuration
```

## Installation & Setup

It is highly recommended to install the framework inside an isolated virtual environment to avoid dependency conflicts.

### 1. Clone the Repository
```bash
git clone [https://github.com/yourusername/core-rl-framework.git](https://github.com/yourusername/core-rl-framework.git)
cd core-rl-framework
```

### 2. Create and Activate a Virtual Environment
You can use either standard Python `venv` or `conda`.

**Using Python `venv` (Linux/macOS):**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Using Conda:**
```bash
conda create -n corerl python=3.10
conda activate corerl
```

### 3. Install Dependencies
Install the required packages from `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 4. Install the Framework in Editable Mode
To use `core_rl` as a library in your other external projects (like your autonomous vehicle repository), install it in editable mode:
```bash
pip install -e .
```

## Quick Start Example

Once installed, you can import and use the framework in any external project:

```python
import gymnasium as gym
from core_rl.agents import SACAgent
from core_rl.buffer import GenericReplayBuffer

# 1. Initialize your environment
env = gym.make("YourCustomEnv-v0")

# 2. Initialize the core-rl brain
agent = SACAgent(
    obs_space=env.observation_space,
    act_space=env.action_space
)
buffer = GenericReplayBuffer(capacity=100000)

# 3. Start training loop
obs, _ = env.reset()
action = agent.select_action(obs)
```

## License

Distributed under the MIT License. See `LICENSE` for more information.