# 🎯 Portfólio de Aprendizagem por Reforço

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![NumPy](https://img.shields.io/badge/NumPy-2.x-013243)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C)
![Práticas](https://img.shields.io/badge/Práticas-PL1--PL9-success)
![Ambientes](https://img.shields.io/badge/Ambientes-5-orange)
![Algoritmos](https://img.shields.io/badge/Algoritmos-20%2B-purple)
![Tests](https://img.shields.io/badge/pytest-71%20passing-brightgreen)
![Grade](https://img.shields.io/badge/Grade-20%2F20-success)
![License](https://img.shields.io/badge/License-Academic-lightgrey)

> **Aprendizagem por Reforço** | Mestrado em Inteligência Artificial | Universidade do Minho | 2025/26

Portefólio completo das Práticas Laboratoriais (PL1–PL9) de Aprendizagem por Reforço — inclui **5 ambientes**, **20+ algoritmos** (bandits, programação dinâmica, predição MC/TD/TD(n), controlo tabular SARSA/Q-Learning/n-step, aproximação linear, policy gradient REINFORCE, planeamento MCTS), **18 scripts executáveis**, **2 notebooks demonstrativos** (um com UI clicável para jogar contra o MCTS) e duas extensões deep RL: **DQN** no Windy Gridworld e **AlphaZero-style** no Tic-Tac-Toe (PUCT MCTS + rede política/valor treinada por self-play). Suite de **71 testes pytest** e **benchmarks reprodutíveis** com saída em JSON + gráficos.

---

## 📊 Resumo dos Algoritmos

### Ambientes Implementados

| Ambiente | Tipo | Prática | Descrição |
|----------|------|:-------:|-----------|
| K-Armed Bandit | Stationary / Non-stationary | PL1 | Testbed multi-armed bandit (Sutton & Barto, Cap. 2) |
| Gridworld | Determinístico / Trap / Estocástico | PL2/PL3 | Grid 4×4 com estados terminais e variantes |
| Blackjack | Card game episódico | PL4 | Simplified Blackjack (Sutton & Barto, Ex. 5.1) |
| Windy Gridworld | Grid com vento por coluna | PL5/PL6 | 7×10 grid com wind strengths variáveis |
| Tic-Tac-Toe | Two-player, self-play | PL6–PL8 | Jogo do galo com encoding 27-dim relativo |

### Algoritmos por Categoria

| Categoria | Algoritmo | Módulo |
|-----------|-----------|--------|
| **Bandits** | ε-Greedy, Decaying ε, UCB, Gradient, Thompson Sampling, Exp3 | `agents/bandits/` |
| **Dynamic Programming** | Policy Evaluation, Value Iteration, Policy Iteration | `agents/dp/` |
| **Prediction** | First-Visit Monte Carlo, TD(0), TD(n) | `agents/prediction/` |
| **Tabular Control** | SARSA, n-step SARSA, MC Control, Q-Learning | `agents/control/` (`q_learning.py`, `sarsa.py`, …) |
| **Function Approximation** | Linear SARSA (NumPy), Torch SARSA (PyTorch), DQN (MLP 64×64 + replay + target net) | `agents/control/` |
| **Policy Gradient** | REINFORCE (Monte Carlo policy gradient, entropy reg.) | `agents/control/reinforce.py` |
| **Model-Based Planning** | MCTS (UCB1 selection, random rollout, backup) | `agents/planning/mcts.py` |
| **MCTS + Neural Net** | AlphaZero-style (PUCT MCTS + policy/value net + self-play) | `agents/planning/alphazero.py` |

### Tic-Tac-Toe — Comparação de Algoritmos

| Algoritmo | Tipo | Win Rate (X) | Losses (X) |
|-----------|------|:------------:|:----------:|
| SARSA | Value-based, on-policy | 98.1% | 13 / 5000 |
| Q-Learning | Value-based, off-policy | 98.7% | **0 / 5000** |
| REINFORCE (vanilla) | Policy gradient, self-play | 78.0% | 218 / 2000 |
| REINFORCE + baseline | Policy gradient + V(s) | **96.0%** | 70 / 2000 |
| MCTS (1000 sims) | Model-based planning, no training | **100.0%** | **0 / 100** |

Q-Learning atinge 0 derrotas (aprende a política ótima diretamente). REINFORCE com baseline (Sutton & Barto, Sec. 13.4) reduz a variância do gradiente subtraindo V(s) aos returns — melhoria de 78% → 96% face ao vanilla REINFORCE. O MCTS com 1000 simulações atinge 100% de win rate como X sem qualquer treino — planeia em tempo de decisão usando o próprio ambiente como modelo.

---

## ✅ Exercícios Opcionais Implementados

| Exercício | Prática | Módulo | Descrição |
|-----------|:-------:|--------|-----------|
| TD(n) Prediction | PL4 | `agents/prediction/td_n.py` | Generalização de TD(0) com n-step returns |
| n-step SARSA | PL5 | `agents/control/n_step_sarsa.py` | Controlo on-policy com janela de n passos |
| Tic-Tac-Toe Environment | PL6 | `envs/tictactoe.py` | Ambiente two-player com self-play |

---

## 📂 Estrutura do Repositório

```
AR1/
├── core/                       # Abstrações genéricas (Environment, Policy, Agent, Episode, Transition)
├── envs/                       # Ambientes de interação
│   ├── bandit.py               #   PL1: K-Armed Bandit (stationary & non-stationary)
│   ├── gridworld.py            #   PL2/PL3: Gridworld (determinístico, trap, estocástico)
│   ├── blackjack.py            #   PL4: Blackjack card game
│   ├── windy_gridworld.py      #   PL5: Windy Gridworld (vento por coluna)
│   └── tictactoe.py            #   PL6–PL8: Tic-Tac-Toe (two-player, self-play)
│
├── mdps/                       # MDPs com modelo conhecido (programação dinâmica)
│   ├── base.py                 #   TabularMDP abstract class
│   ├── gridworld_mdp.py        #   PL2/PL3: Utilidades para Gridworld MDP
│   └── car_rental.py           #   PL3: Jack's Car Rental (Sutton & Barto, Ex. 4.2)
│
├── agents/                     # Algoritmos de aprendizagem
│   ├── bandits/                #   PL1: ε-Greedy, UCB, Gradient, Thompson, Exp3, Decaying ε
│   ├── dp/                     #   PL2/PL3: Policy Eval, Value Iter, Policy Iter
│   ├── prediction/             #   PL4: First-Visit MC, TD(0), TD(n)
│   ├── control/                #   PL5–PL8: SARSA, n-step SARSA, MC Control, Q-Learning,
│   │                           #            Linear SARSA, Torch SARSA, REINFORCE
│   └── planning/               #   PL9: MCTS (Monte Carlo Tree Search)
│
├── features/                   # Feature extractors para aproximação de função
│   ├── windy_gridworld.py      #   PL6: Tile-coding (4 tilings × 24 tiles = 96-dim)
│   └── tictactoe.py            #   PL7: One-hot 27-dim relativo à perspetiva do jogador
│
├── policies/                   # Políticas reutilizáveis
│   ├── blackjack.py            #   ThresholdPolicy para Blackjack
│   └── tictactoe.py            #   random_action, human_policy para Tic-Tac-Toe
│
├── plots/                      # Funções de visualização
├── experiments/                # Rollout, treino, avaliação e helpers
│   ├── tictactoe.py            #   PL7: play_game (avaliar duas políticas head-to-head)
│   ├── reinforce_tictactoe.py  #   PL8: Self-play, vs-random, avaliação REINFORCE
│   └── mcts_tictactoe.py       #   PL9: MCTS vs Random, vs REINFORCE, vs MCTS
│
├── scripts/                    # Scripts executáveis (15 experiências)
│   ├── run_bandits.py              #   PL1: Comparação de 6 algoritmos de bandits
│   ├── run_gridworld.py            #   PL2/PL3: Policy Eval, Value Iter, Policy Iter
│   ├── run_car_rental.py           #   PL3: Jack's Car Rental
│   ├── run_blackjack_prediction.py #   PL4: MC vs TD(0) vs TD(n) + RMSE vs V*
│   ├── run_windy_gridworld_*.py    #   PL5/PL6: SARSA, n-step, MC, Q-Learning, Linear, Torch
│   ├── run_tictactoe.py            #   PL6/PL7: SARSA vs Q-Learning + modo interativo
│   ├── run_reinforce_tictactoe.py  #   PL8: REINFORCE policy gradient + modo interativo
│   └── run_mcts_tictactoe.py       #   PL9: MCTS vs Random / REINFORCE / MCTS + modo interativo
│
├── notebooks/                  # Jupyter notebooks de demonstração
│   └── tictactoe.ipynb             #   PL7+PL8+PL9: Random → REINFORCE → MCTS num único notebook
│
├── tests/                     # Suite pytest (71 testes; 54 sem torch + 17 com torch)
│   ├── test_envs.py            #   PL1-PL9: invariantes dos ambientes
│   ├── test_agents.py          #   PL1-PL9: invariantes dos algoritmos tabulares e lineares
│   ├── test_dqn.py             #   PL6 (extra): DQN — auto-skip sem torch
│   └── test_alphazero.py       #   PL9 (extra): AlphaZero — auto-skip sem torch
│
├── outputs/                    # Gráficos gerados
│   ├── bandits/                #   Epsilon study + battle of the bandits
│   ├── gridworld/              #   Policy eval, value iter, trap, stochastic, policy iter
│   ├── car_rental/             #   Policy iter + value iter (policy & value maps)
│   ├── blackjack_prediction/   #   MC vs TD(0) value surfaces
│   ├── blackjack_control/      #   MC Control vs SARSA policies & values
│   ├── windy_gridworld_*/      #   SARSA, n-step, MC, comparison, linear, torch
│   ├── tictactoe/              #   SARSA vs Q-Learning training & evaluation
│   ├── reinforce_tictactoe/    #   REINFORCE win rates, loss curve, eval summary
│   └── mcts_tictactoe/         #   MCTS vs random sweep, vs REINFORCE, vs MCTS
│
└── RESULTS.md                  # Análise crítica dos resultados (PL1-PL9)
```

---

## 🗺️ Mapa das Práticas

| PL | Tema | Ambientes | Algoritmos | Script |
|:--:|------|-----------|------------|--------|
| 1 | Multi-Armed Bandits | K-Armed Bandit | ε-Greedy, UCB, Gradient, Thompson, Exp3 | `run_bandits` |
| 2 | MDPs & Gridworld | Gridworld | Policy Evaluation, Value Iteration | `run_gridworld` |
| 3 | Dynamic Programming | Gridworld, Car Rental | Policy Iteration, Car Rental DP | `run_gridworld`, `run_car_rental` |
| 4 | Model-Free Prediction | Blackjack | First-Visit MC, TD(0), TD(n) | `run_blackjack_prediction` |
| 5 | Model-Free Control | Windy Gridworld | SARSA, n-step SARSA, MC Control | `run_windy_gridworld_*` |
| 6 | Function Approximation | Windy Gridworld, Tic-Tac-Toe | Linear SARSA, Torch SARSA | `run_windy_gridworld_linear_*` |
| 7 | Tic-Tac-Toe Features | Tic-Tac-Toe | encode_state (27-dim), play_game | `run_tictactoe` |
| 8 | Policy Gradient | Tic-Tac-Toe | REINFORCE (self-play, entropy reg.) | `run_reinforce_tictactoe` |
| 9 | Model-Based Planning | Tic-Tac-Toe | MCTS (UCB1, rollout, backup) | `run_mcts_tictactoe` |

---

## 🚀 Reprodução

### Pré-requisitos

* **Python** 3.10+
* **NumPy** e **Matplotlib**
* **PyTorch** (opcional — apenas para PL6 Torch SARSA)

### Instalação

```bash
# Recomendado — replica o ambiente exato:
pip install -r requirements.txt

# Mínimo — só os scripts não-Torch:
pip install numpy matplotlib

# Opcional (apenas para PL6 Torch SARSA e notebooks):
pip install torch jupyter
```

### Executar Experiências

```bash
# PL1: Multi-Armed Bandits
python -m AR1.scripts.run_bandits --no-show

# PL2/PL3: Gridworld + Dynamic Programming
python -m AR1.scripts.run_gridworld --no-show

# PL3: Jack's Car Rental
python -m AR1.scripts.run_car_rental --no-show

# PL4: Blackjack Prediction (MC vs TD)
python -m AR1.scripts.run_blackjack_prediction --no-show

# PL5: Windy Gridworld — SARSA / n-step SARSA / MC Control
python -m AR1.scripts.run_windy_gridworld_sarsa --no-show
python -m AR1.scripts.run_windy_gridworld_n_step_sarsa --no-show
python -m AR1.scripts.run_windy_gridworld_mc_control --no-show
python -m AR1.scripts.run_windy_gridworld_comparison --no-show

# PL6: Function Approximation — Linear TD / Linear SARSA / Torch SARSA
python -m AR1.scripts.run_windy_gridworld_linear_td --no-show
python -m AR1.scripts.run_windy_gridworld_linear_sarsa --no-show
python -m AR1.scripts.run_windy_gridworld_torch_sarsa --no-show

# PL5: Windy Gridworld — Q-Learning (off-policy, comparável ao SARSA)
python -m AR1.scripts.run_windy_gridworld_q_learning --no-show

# Extra: DQN (MLP 64×64 + replay buffer + target net) vs Linear SARSA
python -m AR1.scripts.run_windy_gridworld_dqn --no-show

# Extra: AlphaZero-style (PUCT MCTS + policy/value net via self-play)
python -m AR1.scripts.run_alphazero_tictactoe --no-show

# Suite de benchmarks — corre algoritmos chave, mede métricas, gera JSON e gráficos
python -m AR1.scripts.run_benchmarks --no-show
# Resultados em outputs/benchmarks/{benchmarks.json, windy_summary.png, tictactoe_summary.png, ...}

# Notebook unificado Random -> SARSA -> Q-Learning -> REINFORCE -> MCTS
jupyter notebook notebooks/demo.ipynb

# PL6/PL7: Tic-Tac-Toe — SARSA vs Q-Learning
python -m AR1.scripts.run_tictactoe --no-show

# PL8: Tic-Tac-Toe — REINFORCE policy gradient (self-play)
python -m AR1.scripts.run_reinforce_tictactoe --no-show

# PL9: Tic-Tac-Toe — MCTS (planning, no training)
python -m AR1.scripts.run_mcts_tictactoe --no-show
```

### 🎮 Jogar contra os Agentes Treinados

```bash
# Jogar contra Q-Learning (0 derrotas na avaliação)
python -m AR1.scripts.run_tictactoe --play

# Jogar contra SARSA
python -m AR1.scripts.run_tictactoe --play --play-algo sarsa

# Jogar contra REINFORCE (policy gradient)
python -m AR1.scripts.run_reinforce_tictactoe --play

# Jogar contra MCTS (model-based planning, sem treino)
python -m AR1.scripts.run_mcts_tictactoe --play
```

> Os gráficos são guardados em `AR1/outputs/`. Remover `--no-show` para exibir interativamente.

---

## ✅ Suite de Testes

71 testes `pytest` (54 sem PyTorch + 17 com) cobrem ambientes e algoritmos — garantia de que as invariantes
(forma das features, recompensas, dinâmica do vento, deteção de vitória, gradientes
da política, etc.) se mantêm corretas após qualquer alteração.

```bash
# A partir da raiz do repositório (pasta-pai de AR1/):
PYTHONPATH=. pytest AR1/tests -q
# Esperado: "71 passed" (ou "54 passed, 17 skipped" se PyTorch não estiver instalado)
```

Cobertura por módulo:
* `tests/test_envs.py` — KArmedBandit, Gridworld, GridworldTrap, transições estocásticas, Blackjack, Windy Gridworld, TicTacToe.
* `tests/test_agents.py` — 6 bandits, DP (Policy Eval, VI, PI), Predição (MC/TD/TDn), Controlo tabular (SARSA, Q-Learning, n-step SARSA, MC Control), Aproximação Linear, Features TicTacToe, REINFORCE, MCTS.
* `tests/test_dqn.py` — Deep Q-Network (rede, replay buffer, target net, ε-decay). Auto-*skip* se PyTorch não estiver instalado.
* `tests/test_alphazero.py` — AlphaZero-style (rede política/valor, PUCT MCTS, *visit distribution*, treino). Auto-*skip* se PyTorch não estiver instalado.

---

## 📊 Análise de Resultados

O ficheiro [`RESULTS.md`](RESULTS.md) contém uma discussão crítica completa das experiências:
contextualização teórica de cada PL, resultados numéricos principais, comparações
entre algoritmos e decisões técnicas transversais.

---

## 👤 Autor

| Nome | Nº | Email |
|------|----|-------|
| Luís Miguel Pereira Silva | PG60390 | pg60390@alunos.uminho.pt |

---

## 📜 Licença

Este trabalho é de cariz estritamente académico. Universidade do Minho, Escola de Engenharia, Departamento de Informática.
