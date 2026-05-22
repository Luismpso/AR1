from .base import ControlAgent
from .monte_carlo import MonteCarloControl
from .n_step_sarsa import NStepSarsaControl
from .q_learning import QLearningControl
from .sarsa import SarsaControl

# Function approximation
try:
    from .linear_sarsa import LinearSarsaControl
except ImportError:
    pass

# Torch-based agents are optional (only available when torch is installed).
try:
    from .torch_sarsa import TorchSarsaControl
    from .dqn import DQNControl
except ImportError:
    pass
