# Análise de Resultados — Portefólio de Aprendizagem por Reforço

> **Aprendizagem por Reforço** · Mestrado em Inteligência Artificial · Universidade do Minho · 2025/26
> **Aluno:** Luís Miguel Pereira Silva (PG60390)
> **Entrega:** 26/05/2026 23h59 · **Defesa:** 09/06/2026

Este documento sintetiza os resultados experimentais obtidos nas Práticas
Laboratoriais **PL1–PL9** e discute as escolhas técnicas. Todos os
gráficos estão em `outputs/<experiência>/` e podem ser regerados com os
scripts em `scripts/run_*.py`.

> **Reprodutibilidade.** Todos os scripts aceitam `--seed`. A suite
> `pytest AR1/tests` (54 testes base + 18 testes em torch totalizando 72)
> garante invariantes dos ambientes e algoritmos.

---

## Mapa rápido das 9 Práticas

| PL | Tema | Ambiente | Algoritmos | Script |
|:--:|------|----------|------------|--------|
| 1 | Multi-Armed Bandits | K-Armed Bandit | ε-Greedy, ε-decay, UCB, Gradient, Thompson, Exp3 | `run_bandits` |
| 2 | MDPs & Gridworld | Gridworld | Policy Evaluation, Value Iteration | `run_gridworld` |
| 3 | Dynamic Programming | Gridworld, Car Rental | Policy Iteration, Car Rental DP | `run_gridworld`, `run_car_rental` |
| 4 | Model-Free Prediction | Blackjack | First-Visit MC, TD(0), **TD(n)** | `run_blackjack_prediction` |
| 5 | Model-Free Control | Windy Gridworld | SARSA, **n-step SARSA**, MC Control | `run_windy_gridworld_*` |
| 6 | Function Approximation | Windy Gridworld, Tic-Tac-Toe | Linear SARSA, Torch SARSA, **DQN** | `run_windy_gridworld_linear_*`, `run_windy_gridworld_dqn` |
| 7 | Tic-Tac-Toe Features | Tic-Tac-Toe | encode_state (27-dim), play_game | `run_tictactoe` |
| 8 | Policy Gradient | Tic-Tac-Toe | REINFORCE (vanilla + baseline) | `run_reinforce_tictactoe` |
| 9 | Model-Based Planning | Tic-Tac-Toe | MCTS, **AlphaZero-style (PUCT + rede)** | `run_mcts_tictactoe`, `run_alphazero_tictactoe` |

Itens a **negrito** = exercícios opcionais / extensões extra.

---

## PL1 — Multi-Armed Bandits

**Ambiente:** `KArmedBandit` (Sutton & Barto, Cap. 2) — 10 braços com
`q_true ~ N(0, 1)` e recompensas `r ~ N(q_true[a], 1)`. Suporta modo
não-estacionário (passeio aleatório em `q_true`).

**Algoritmos comparados:** ε-Greedy, ε-Greedy com *decay*, UCB1,
Gradient Bandit, Thompson Sampling, Exp3.

**Resultados** (ver `outputs/bandits/all_algorithms.png` e
`epsilon_greedy_study.png`):

| Algoritmo | Comportamento observado |
|-----------|--------------------------|
| ε-Greedy (ε=0.1) | Bom *baseline*; saturação em ~80 % de ações ótimas |
| ε-Greedy decay | Bate ε constante em ambientes estacionários |
| UCB1 | Convergência muito rápida; melhor curva inicial |
| Thompson Sampling | Topo da tabela em recompensa média a longo prazo |
| Gradient Bandit | Mais sensível à escolha de α |
| Exp3 | Pior em estacionário (foi desenhado para *adversarial*) |

**Discussão.** Os resultados confirmam a intuição teórica:
estratégias *Bayesianas* (Thompson) e *otimistas* (UCB1) ganham por
equilibrarem exploração e exploração sem decisões aleatórias cegas.

---

## PL2/PL3 — Programação Dinâmica em Gridworld + Car Rental

**Ambientes Gridworld:**
* `Gridworld` 4×4 determinístico (Sutton & Barto, Ex. 4.1).
* `GridworldTrap` — célula (1, 2) custa −10 (variante penalizada).
* Versão estocástica — 0.8 de probabilidade da ação intendida, 0.1 para
  cada *slip* perpendicular.

**Algoritmos:** Policy Evaluation, Value Iteration, Policy Iteration,
Stochastic Value Iteration — comparados em γ ∈ {0.5, 0.9, 0.99}.

**Resultados** (ver `outputs/gridworld/`):

* Policy Evaluation para a política uniforme converge em **118 iterações**
  (θ = 1e-6) e reproduz exatamente os valores do manual.
* Value Iteration converge em **4 iterações** — confirma a vantagem
  face a Policy Iteration (3 melhorias × 118 avaliações).
* `GridworldTrap` força a política ótima a desviar-se da célula (1, 2).
* A variante estocástica produz políticas mais afastadas de paredes
  e armadilhas — efeito da incerteza.

**Jack's Car Rental (`outputs/car_rental/`):** ambos PI e VI convergem
para a mesma política ótima (verificado por comparação `pi_policy.png`
vs `vi_policy.png`). O *cache* `(loc_id, cars_after_move) → (P, E[rent])`
torna o cálculo exato tratável (~5 s).

---

## PL4 — Predição Model-Free no Blackjack

**Ambiente:** `BlackjackEnv` (Ex. 5.1) — estado = (soma do jogador,
carta do dealer, ás utilizável). Política fixa `ThresholdPolicy(stick_at=20)`.

**Algoritmos:** First-Visit Monte Carlo, TD(0), **TD(n)** (exercício
opcional implementado em `agents/prediction/td_n.py`).

**Resultados** (ver `outputs/blackjack_prediction/`):

* As superfícies V(s) do MC e do TD(0) são visualmente quase idênticas
  após ~200 000 episódios.
* TD(n=4) atinge RMSE comparável ao MC com **menos episódios** — ganho
  claro do *bootstrap* multi-step.
* TD(n=1) reproduz exatamente TD(0) (testado por `test_tdn_with_n1_matches_td0`).

**Discussão.** TD(n) materializa o trade-off *bias-variance*:
n = 1 → mais *bias*, menos variância; n = ∞ ≡ MC → sem *bias*, mais
variância. n = 4 é um ponto intermédio bom para este problema.

---

## PL5 — Controlo Tabular no Windy Gridworld

**Ambiente:** `WindyGridworldEnv` 7×10, vento por coluna
`(0, 0, 0, 1, 1, 1, 2, 2, 1, 0)`, recompensa −1 por passo.

**Algoritmos:** SARSA, **n-step SARSA (n=4)** (opcional), MC Control,
Q-Learning.

**Resultados (`outputs/benchmarks/benchmarks.json`, 300 episódios, seed 7):**

| Algoritmo | Episódios até convergir | Comprimento greedy final | Tempo treino |
|-----------|------------------------:|-------------------------:|-------------:|
| SARSA              | 195 | 100 | 0.08 s |
| Q-Learning         | 146 | **15** | 0.08 s |
| n-step SARSA (n=4) | **97** | 20 | 0.10 s |

(Convergência = janela deslizante de 20 episódios com média ≤ 20 passos.)

**Discussão.**
* **n-step SARSA bate SARSA(1) em sample efficiency** — propaga sinal
  multi-step. A 300 episódios já tem política perto da ótima.
* **Q-Learning encontra a política ótima** (15 passos) — *off-policy*
  não fica preso à exploração ε.
* **SARSA(1)** ainda não convergiu aos 300 ep (path = 100), mas com
  500 ep estabiliza em ~17 passos.

---

## PL6 — Aproximação de Função

### Linear (tile-coding)

**Features:** 4 *tilings* × 24 *tiles* = **96 dims por estado**; encoding
*action-specific* dá 384 dims por (s, a). Cada *tile* contribui 1/4 para
manter a soma das ativações constante.

**Algoritmos:** Linear TD(0) (semi-gradient), Linear SARSA (NumPy),
Torch SARSA (PyTorch — modo manual via *autograd* e modo *optimizer*).

**Resultados** (`outputs/windy_gridworld_linear_sarsa/`, `_torch_sarsa/`):

* O *value heatmap* do Linear SARSA é coerente com o efeito do vento.
* NumPy ≡ PyTorch — confirma que `loss = ½(pred − target)² ⇒
  grad = (pred − target) · φ` é exatamente `w += α · δ · φ`.
* TD-errors decaem suavemente com α = 0.05.

### Extra — DQN

**Algoritmo:** Deep Q-Network (Mnih et al., 2015) com **replay buffer 10k**,
**target network** sincronizada a cada 100 *updates*, ε de 1.0 → 0.05 em
5000 passos, Adam (lr = 5e-4), *gradient clipping* a 10.0.

**Arquitetura:** MLP `(2 → 64 → 64 → 4)` com ReLU. Input: `(row, col)`
normalizado para `[0, 1]²` — a rede aprende as features sozinha, sem
tile-coding.

**Comparação direta** (`outputs/windy_gridworld_dqn/lengths_comparison.png`):

| Critério | Linear SARSA (tile coding) | DQN (MLP 64×64) |
|----------|----------------------------|-----------------|
| Features de entrada | 384-dim sparse | 2-dim densa |
| Parâmetros | ≈ 384 lineares | ≈ 4 800 não-lineares |
| Convergência (300 ep) | estabiliza em ~20 passos | oscila 200–500 passos |
| Estabilidade | muito estável | requer target net + clipping |

**Discussão.** O DQN demonstra a abstração *end-to-end* típica do deep
RL — aprende sem features manuais. **Em troca paga em sample efficiency.**
Para um problema deste tamanho, Linear SARSA + tile coding ganha
claramente. O DQN brilha quando o espaço de estados é grande demais para
tile-coding (continuous control, imagens).

---

## PL7 — Tic-Tac-Toe: Features e Avaliação Head-to-Head

**Ambiente:** `TicTacToeEnv` (implementado para o portefólio — exercício
do enunciado da PL6). Tabuleiro 3×3, X = +1, O = −1.

**Features:** `encode_state(board, current_player)` produz vetor one-hot
27-dim **relativo à perspetiva do jogador atual**. Esta escolha é
fundamental — o mesmo agente joga X **ou** O sem retreinar.

**Resultados** (`outputs/tictactoe/`):

| Algoritmo | Win Rate (X) | Empates | Derrotas (em 5000) |
|-----------|:------------:|:-------:|:-------------------:|
| Random vs Random   | ~58 % | ~12 % | — |
| SARSA              | 98.1 % | 1.6 % | 13 |
| Q-Learning         | **98.7 %** | 1.3 % | **0** |

**Discussão.** Q-Learning atinge **0 derrotas em 5000 jogos** porque
aprende a política ótima off-policy. SARSA fica conservador por
incorporar a exploração ε no alvo.

---

## PL8 — Policy Gradient (REINFORCE)

**Algoritmos:** REINFORCE vanilla e REINFORCE com **baseline V(s)**
(Sutton & Barto, Sec. 13.4) — reduz variância sem introduzir *bias*.

Treino: *self-play* + 30 % vs aleatório, 4000 episódios.

**Resultados** (`outputs/reinforce_tictactoe/`):

| Variante | Win Rate (X) | Derrotas (em 2000) |
|----------|:------------:|:-------------------:|
| Vanilla REINFORCE   | 78.0 % | 218 |
| REINFORCE + baseline | **96.0 %** | 70 |

**Discussão.** O ganho 78 % → 96 % confirma o resultado clássico: a
variância do gradiente vanilla é o gargalo. Subtrair `V(s_t)` aos
retornos `G_t` resolve isto sem alterar a esperança do estimador.

---

## PL9 — Planeamento Model-Based (MCTS)

**Algoritmo:** MCTS clássico — seleção UCB1, *rollout* aleatório,
*backup* invertendo sinal. Ação escolhida = filho mais visitado.

**Resultados** (`outputs/mcts_tictactoe/`):

| Adversário | Simulações | Win Rate (X) |
|------------|:----------:|:------------:|
| Random | 100 | 90 % |
| Random | 500 | 99 % |
| Random | 1000 | **100 %** |
| Random | 200 (benchmarks) | **99.5 %** |
| REINFORCE com baseline | 1000 | ~70 % vence |
| MCTS(50) | MCTS(1000) | 0 % vence (planeador forte impõe-se) |

**Discussão.** MCTS com 1000 simulações por jogada atinge **100 %**
vs aleatório **sem qualquer treino** — está a usar o ambiente como modelo
perfeito. Mais simulações ⇒ jogadas mais informadas; dois MCTS fortes
acabam em empate (resultado teórico do jogo).

### Extra — AlphaZero-style

**Componentes:**
* **Rede política+valor** `27 → 64 → 64 → {9 logits, 1 value tanh}`.
* **PUCT MCTS** — a folha não-terminal é avaliada pela rede `(π, v)`
  em vez de *rollout* aleatório; seleção usa
  `argmax_a [Q + c_puct · P(a|s) · √ΣN / (1 + N)]`.
* ***Self-play*** com **Dirichlet noise** (α=0.3, ε=0.25) na raiz e
  amostragem por temperatura nas primeiras 4 jogadas.
* **Treino** com perda combinada `L = MSE(v, z) + CE(π_pred, π_visits) +
  weight_decay`.

**Resultados** (`outputs/alphazero_tictactoe/evaluation.png`,
4 iterações × 20 self-play × 32 sims):

| Adversário | Como X | Como O |
|------------|:------:|:------:|
| Random     | **98 % win** | **94 % win** |
| MCTS-200 (clássico)  | 15 % win, 85 % draw | 95 % draw |
| REINFORCE (baseline) | **100 % win** | **100 % win** |

**Discussão.** Mesmo com orçamento didático (4 iterações), o AlphaZero
domina o REINFORCE (100 %) e arranca empates consistentes contra um
MCTS-200 — sinal de que a rede de valor já direciona a procura mesmo
com poucas simulações (32 vs 200 do MCTS clássico). Em Go ou Xadrez,
escalava para ResNets + milhões de self-play games.

---

## Engenharia e Reprodutibilidade

### Decisões transversais

| Decisão | Motivação |
|---------|-----------|
| Pacote modular (`envs/`, `agents/`, `policies/`, `experiments/`) | Estrutura proposta na PL4; troca de componentes trivial |
| `Transition` e `Episode` em `core/base.py` | Interface uniforme para todos os agentes |
| Encoding TicTacToe relativo ao jogador | Mesmo agente joga X e O |
| Tile-coding em vez de features brutas | Necessário para capturar o vento com modelo linear |
| Cache no Car Rental | Reduz cálculo de O(20² × 9² × 11²) para tempo razoável |
| `--no-show` em todos os scripts | Permite correr em CI / batch sem display |
| `pytest AR1/tests` | Garante invariantes mesmo após *refactorings* |
| Seeds em todos os scripts | Resultados reproduzíveis |

### Suite de Testes (`tests/`)

**72 testes pytest** (54 base + 8 DQN + 10 AlphaZero — estes últimos
auto-skip se torch não estiver instalado):

* `test_envs.py` — 24 testes para KArmedBandit, Gridworld (deterministico,
  trap, estocástico), Blackjack, Windy Gridworld, TicTacToe.
* `test_agents.py` — 30 testes para os 6 bandits, DP (PE/VI/PI),
  predição (MC/TD/TDn), controlo tabular (SARSA, Q-Learning, n-step
  SARSA, MC Control), aproximação linear, features TicTacToe, REINFORCE,
  MCTS.
* `test_dqn.py` — 8 testes (shape da rede, replay buffer, target sync,
  ε-decay, treino).
* `test_alphazero.py` — 10 testes (rede, PUCT, *visit distribution*,
  iteração de treino).

```bash
PYTHONPATH=. python -m pytest AR1/tests -q
# Esperado: 72 passed in ~30s  (ou 54 passed, 18 skipped sem torch)
```

### Suite de Benchmarks (`scripts/run_benchmarks.py`)

Mede para cada algoritmo: **tempo de treino**, **episódios até
convergir** (janela deslizante de 20 com média ≤ 20 passos), e
**métrica final** (caminho greedy ou win-rate). Exporta
`outputs/benchmarks/benchmarks.json` + 3 figuras
(`windy_summary.png`, `windy_curves.png`, `tictactoe_summary.png`).

### Notebooks de demonstração

* `notebooks/tictactoe.ipynb` (30 células) — PL7→PL9 num único fluxo:
  Random → REINFORCE (com curvas de treino) → MCTS, com **UI
  interativa** (botões clicáveis ipywidgets) para jogar contra o MCTS.
* `notebooks/demo.ipynb` (17 células) — comparação executável de
  Random → SARSA → Q-Learning → REINFORCE → MCTS na mesma sessão,
  com gráfico final empilhado win/draw/loss separado por X e O.

---

## Reprodução Rápida

```bash
# Smoke test (≈ 1 s)
PYTHONPATH=. python -m pytest AR1/tests -q

# Todas as experiências (~10–15 min em CPU modesto)
for s in bandits gridworld car_rental blackjack_prediction \
         windy_gridworld_sarsa windy_gridworld_q_learning \
         windy_gridworld_n_step_sarsa windy_gridworld_mc_control \
         windy_gridworld_comparison \
         windy_gridworld_linear_sarsa windy_gridworld_linear_td \
         windy_gridworld_torch_sarsa windy_gridworld_dqn \
         tictactoe reinforce_tictactoe mcts_tictactoe \
         alphazero_tictactoe benchmarks; do
  python -m AR1.scripts.run_$s --no-show
done
```

---

## Conclusão — o que cada PL adiciona

| Capacidade adicionada | PL | Algoritmo de referência | Ganho típico em Tic-Tac-Toe |
|-----------------------|:--:|--------------------------|------------------------------|
| Multi-armed exploration | 1 | UCB / Thompson | — |
| Modelo conhecido       | 2/3 | Value Iteration | política ótima em 4 iter |
| Sem modelo, prediz V   | 4 | MC, TD(n) | superfícies V(s) precisas |
| Aprende Q(s,a)         | 5 | Q-Learning | 0 derrotas em 5000 jogos |
| Generaliza por features| 6 | Linear SARSA / DQN | resolve grids contínuos |
| Política estocástica   | 8 | REINFORCE + baseline | 96 % vs random |
| Planeia em tempo real  | 9 | MCTS | 100 % sem treino |
| Política guiada por rede | 9 (extra) | AlphaZero | 100 % vs REINFORCE; empata MCTS |

Cada algoritmo *acrescenta* uma capacidade e o ganho é mensurável nas
tabelas acima.

---

*Documento gerado para a submissão do portefólio individual (26/05/2026).*
*Autor: Luís Miguel Pereira Silva (PG60390) · luimpsoo@gmail.com*
