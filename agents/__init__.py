# Agents organized by task:
#   - bandits/     → PL1
#   - dp/          → PL2/PL3
#   - prediction/  → PL4
#   - control/     → PL5
from .control.monte_carlo import MonteCarloControl
from .control.n_step_sarsa import NStepSarsaControl
from .control.sarsa import SarsaControl
from .prediction.monte_carlo import FirstVisitMonteCarloPrediction
from .prediction.td import TD0Prediction
