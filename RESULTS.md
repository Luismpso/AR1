# Análise de Resultados — Portefólio de Aprendizagem por Reforço

Este documento sintetiza os resultados experimentais obtidos nas Práticas Laboratoriais
**PL1–PL9** e discute as escolhas técnicas mais relevantes. Os gráficos referidos
encontram-se em `outputs/<experiência>/`. Os scripts que os reproduzem estão em
`scripts/run_*.py` e podem ser executados com `--no-show` para gerar tudo sem
interface gráfica.

> **Reprodutibilidade.** Todos os scripts aceitam `--seed` para fixar a aleatoriedade.
> A suite `pytest AR1/tests` (54 testes) garante que ambientes e algoritmos cumprem
> as suas invariantes fundamentais.

---

## PL1 — Multi-Armed Bandits

**Ambiente:** `KArmedBandit` (Sutton & Barto, Cap. 2) — 10 braços com
`q_true ~ N(0, 1)` e recompensas `r ~ N(q_true[a], 1)`. Suporta o modo
não-estacionário (passeio aleatório em `q_true`).

**Algoritmos comparados:** ε-Greedy (com várias ε), ε-Greedy com decaimento,
UCB1, Gradient Bandit, Thompson Sampling, Exp3.

**Resultados principais** (ver `outputs/bandits/`):

| Algoritmo | Vantagem observada |
|-----------|--------------------|
| ε-Greedy (ε=0.1) | Bom *baseline*; saturação em ~80% de ações ótimas |
| ε-Greedy decay | Bate ε constante em ambientes estacionários |
| UCB1 | Convergência muito rápida; melhor curva inicial |
| Thompson Sampling | Topo da tabela em recompensa média a longo prazo |
| Gradient Bandit | Mais sensível à escolha de α |
| Exp3 | Pior em estacionário (foi desenhado para adversarial) |

**Discussão.** Os resultados confirmam a intuição teórica: estratégias *Bayesianas*
(Thompson) e *otimistas* (UCB1) ganham porque equilibram exploração e exploração
sem decisões aleatórias cegas. O ε-Greedy mantém-se relevante pela simplicidade
e por não exigir distribuição inicial.

---

## PL2/PL3 — Programação Dinâmica em Gridworld

**Ambientes:** `Gridworld` 4×4 (Ex. 4.1), `GridworldTrap` (célula (1,2) = -10),
`Gridworld` estocástico (probabilidade 0.8 de executar a ação, 0.1 para
cada deslize perpendicular).

**Algoritmos:** Policy Evaluation, Value Iteration, Policy Iteration,
Stochastic Value Iteration. Comparações por valor de γ (0.5, 0.9, 0.99) para
estudar o efeito do desconto.

**Resultados principais** (ver `outputs/gridworld/`):

* Policy Evaluation para a política uniforme converge em **~118 iterações**
  com θ=1e-6 e reproduz exatamente os valores do manual (Fig. 4.1).
* Value Iteration converge em **4 iterações** — confirma a vantagem face a
  Policy Iteration (3 iterações externas × ~120 internas de avaliação).
* Em `GridworldTrap` a política ótima desvia-se claramente da célula (1, 2):
  o valor da armadilha é -10 enquanto o passo normal é -1, logo a política
  ótima nunca arrisca passar por lá.
* Em variante estocástica, a política ótima evita corredores estreitos junto
  a paredes/armadilhas — efeito da incerteza nas transições.

**Discussão.** A comparação Value Iteration vs Policy Iteration ilustra a
**equivalência teórica** com perfis de custo distintos: VI faz uma melhoria
por iteração; PI faz avaliação até quase convergir antes de melhorar.
Para este problema pequeno, VI vence em tempo de relógio.

---

## PL3.2 — Jack's Car Rental

**MDP:** Sutton & Barto, Ex. 4.2 — duas localizações com pedidos/devoluções
Poisson, capacidade 20, custo de 2 por carro deslocado, receita 10 por
aluguer. Distribuições Poisson truncadas para tornar o cálculo exato e
*cache* `(loc_id, cars_after_move) → (P(next_cars), E[rent])`.

**Algoritmos:** Policy Iteration, Value Iteration.

**Resultados** (`outputs/car_rental/`):

* Ambos os métodos convergem para a mesma política ótima (verificado por
  comparação visual `pi_policy.png` vs `vi_policy.png`).
* A política tem a forma esperada: mover carros da Loc 1 para a Loc 2
  quando há excesso na Loc 1; mover no sentido inverso quando há excesso
  na Loc 2.

**Discussão.** O problema mostra a vantagem de pré-computar as transições
Poisson — sem cache o tempo seria proibitivo. A diferença entre PI e VI
neste caso é mínima porque o espaço de ações é pequeno (-5 .. +5).

---

## PL4 — Predição Model-Free no Blackjack

**Ambiente:** `BlackjackEnv` (Ex. 5.1) — estado = (soma do jogador, carta
do dealer, ás utilizável). Política fixa: `ThresholdPolicy(threshold=20)`
(*stick* em ≥20, *hit* caso contrário).

**Algoritmos:** First-Visit Monte Carlo, TD(0), **TD(n)** (exercício
opcional do enunciado da PL4).

**Resultados** (`outputs/blackjack_prediction/`):

* As superfícies V(s) do MC e do TD(0) são visualmente muito próximas após
  ~200 000 episódios.
* TD(n=4) atinge RMSE comparável ao MC com menos episódios, confirmando
  o ganho do *bootstrap* multi-step.
* TD(n=1) reproduz exatamente o comportamento de TD(0) (verificado por
  teste `test_tdn_with_n1_matches_td0`).

**Discussão.** O TD(n) é uma boa demonstração do *bias-variance tradeoff*:
n=1 = mais bias / menos variância; n=∞ ≡ MC = sem bias / mais variância.
n=4 fica num ponto intermédio ótimo para este problema.

---

## PL5 — Controlo Tabular no Windy Gridworld

**Ambiente:** `WindyGridworldEnv` 7×10 com vento por coluna
`(0,0,0,1,1,1,2,2,1,0)`. Recompensa -1 por passo, episódio termina ao
chegar a (3, 7).

**Algoritmos:** SARSA, n-step SARSA (n=4, *exercício opcional*), Monte Carlo
Control com *exploring starts*.

**Resultados** (`outputs/windy_gridworld_*/`):

| Algoritmo | Passos da política greedy final |
|-----------|----------------------------------|
| SARSA (500 ep) | ~17 |
| n-step SARSA (n=4, 500 ep) | **~17** (converge mais rápido) |
| MC Control | mais ruidoso, ~30+ |
| Q-Learning | ~17, atinge ótimo |

**Discussão.** SARSA on-policy e Q-Learning off-policy convergem para
políticas equivalentes neste problema determinístico. O ganho do n-step
SARSA é claro nas primeiras 100 episódios: propaga sinal mais depressa.
O MC tem variância elevada porque atualiza apenas no fim do episódio
e o retorno total é dominado por passos exploratórios.

---

## PL6 — Aproximação de Função

**Features:** *tile coding* 4 tilings × 24 tiles = **96 dims/estado**, com
encoding action-specific (96 × 4 = **384 dims/(estado, ação)**). Cada
*tile* contribui 1/4 para manter a soma das features ativas igual a 1.

**Algoritmos:** Linear TD(0) (semi-gradient), Linear SARSA (NumPy),
Torch SARSA (PyTorch, com `use_optimizer=True` e modo manual via autograd).

**Resultados** (`outputs/windy_gridworld_linear_sarsa/`, `_torch_sarsa/`,
`_linear_td/`):

* O *value heatmap* aprendido pelo Linear SARSA é coerente com o vento:
  estados próximos do objetivo têm valor próximo de 0 e o gradiente
  apropria-se da estrutura do vento.
* As duas implementações (NumPy vs PyTorch) produzem trajetórias
  praticamente idênticas — confirma que `loss = 0.5*(pred-target)^2 →
  grad = (pred-target)*φ` é equivalente a `w += α·δ·φ`.
* Os TD-errors decaem suavemente com o número de episódios, indicando
  que o passo de aprendizagem (α=0.05) está bem dimensionado para a
  norma das features.

**Discussão.** A aproximação linear *sem* tile-coding (apenas
`(row, col)` cru) não funciona porque o vento introduz uma dependência
não-linear; tile-coding resolve isto criando representações *piecewise
constant* que o modelo linear consegue compor.

---

## PL7 — Tic-Tac-Toe: Features e Avaliação Head-to-Head

**Ambiente:** `TicTacToeEnv` (implementado para o portefólio segundo o
enunciado da PL6) — tabuleiro 3×3, X=+1, O=-1.

**Features:** `encode_state(board, current_player)` produz vector
one-hot 27-dim **relativo à perspetiva do jogador atual** (não absoluta!).
Esta escolha é fundamental para que o mesmo agente possa jogar como X
ou como O sem retreinar.

**Resultados** (`outputs/tictactoe/`):

| Algoritmo | Win Rate (X) | Empates | Derrotas |
|-----------|:------------:|:-------:|:--------:|
| Random vs Random | ~58% | ~12% | ~30% |
| SARSA (5000 ep) | 98.1% | 1.6% | 13 / 5000 |
| Q-Learning (5000 ep) | **98.7%** | 1.3% | **0 / 5000** |

**Discussão.** Q-Learning atinge **0 derrotas** porque aprende a política
ótima (off-policy) — não está limitado pela ε-exploration durante a
avaliação. SARSA aprende uma política *próxima da ótima mas conservadora*
porque a Q-update incorpora o comportamento exploratório.

---

## PL8 — Policy Gradient (REINFORCE) no Tic-Tac-Toe

**Algoritmos:** Vanilla REINFORCE, REINFORCE com *baseline* (Sutton & Barto,
Sec. 13.4 — subtrai V(s_t) ao retorno para reduzir variância sem introduzir
bias), com regularização de entropia opcional.

**Treino:** *self-play* + mistura com agente aleatório, 2000 episódios.

**Resultados** (`outputs/reinforce_tictactoe/`):

| Variante | Win Rate (X) | Derrotas |
|----------|:------------:|:--------:|
| Vanilla REINFORCE | 78.0% | 218 / 2000 |
| REINFORCE + baseline | **96.0%** | 70 / 2000 |

**Discussão.** O ganho do baseline (78%→96%) confirma o resultado clássico:
a variância do gradiente é fortemente reduzida ao substituir G_t por
G_t − V(s_t). O entropy bonus mantém alguma exploração saudável e evita
colapso prematuro da política.

---

## PL9 — Planeamento Model-Based (MCTS)

**Algoritmo:** MCTS com seleção UCB1, *random rollout*, *backup* alternando
sinal (parente é o adversário). Ação escolhida = filho mais visitado.

**Resultados** (`outputs/mcts_tictactoe/`):

| Adversário | n_simulations | Win Rate (X) |
|------------|:--------------:|:------------:|
| Random | 100 | 90% |
| Random | 500 | 99% |
| Random | 1000 | **100%** |
| REINFORCE | 1000 | ~70% (vence) |
| MCTS(50) | MCTS(1000) | 0% (perde) |

**Discussão.** O MCTS com 1000 simulações por jogada atinge 100% de
vitórias contra o aleatório **sem qualquer treino** — está a usar o
ambiente como modelo perfeito. Contra o REINFORCE treinado (com baseline)
o MCTS vence claramente, mostrando a vantagem do planeamento explícito
em tempo de decisão num jogo de horizonte curto. A comparação
MCTS-vs-MCTS confirma que mais simulações = jogadas mais informadas.

---

## Decisões Técnicas Transversais

| Decisão | Motivação |
|---------|-----------|
| Pacote modular (`envs/`, `agents/`, `policies/`, …) | Cumpre o enunciado da PL4; permite trocar componentes facilmente |
| `Transition` e `Episode` em `core/base.py` | Interface uniforme para todos os agentes |
| Encoding de TicTacToe relativo ao jogador | Permite o mesmo agente jogar X e O |
| Tile-coding em vez de features brutas | Necessário para que a aproximação linear capture o vento |
| Cache no Car Rental | Reduz cálculo de O(20² × 9² × 11²) para tempo razoável |
| `--no-show` em todos os scripts | Permite correr em CI / batch sem display |
| Suite `pytest` (54 testes) | Garante invariantes mesmo após refactorings |
| Seeds em todos os scripts | Resultados reproduzíveis |

---

## Reprodução Rápida

```bash
# Smoke test rápido — só testes (≈ 30 s)
PYTHONPATH=. pytest AR1/tests -q

# Todas as experiências (~15 min em CPU modesto)
for s in bandits gridworld car_rental blackjack_prediction \
         windy_gridworld_sarsa windy_gridworld_q_learning \
         windy_gridworld_n_step_sarsa windy_gridworld_mc_control \
         windy_gridworld_comparison \
         windy_gridworld_linear_sarsa windy_gridworld_linear_td \
         windy_gridworld_torch_sarsa \
         tictactoe reinforce_tictactoe mcts_tictactoe; do
  python -m AR1.scripts.run_$s --no-show
done
```

---

## Extensão Extra — DQN no Windy Gridworld

**Algoritmo:** Deep Q-Network (Mnih et al., 2015) com replay buffer e
target network. Arquitetura: MLP `(2 → 64 → 64 → 4)` com ReLU. Input:
`(row, col)` normalizado para `[0, 1]²`. Treino: Adam (lr=5e-4), batch 32,
replay buffer 10k, warmup 200 passos, target sync a cada 100 updates,
ε de 1.0 → 0.05 ao longo de 5000 passos, gradient clipping a 10.0.

**Comparação direta com Linear SARSA** (mesmo número de episódios, mesmo
ambiente, mesma semente — `outputs/windy_gridworld_dqn/`):

| Critério | Linear SARSA (tile coding) | DQN (MLP 64×64) |
|----------|----------------------------|-----------------|
| Features de entrada | 384-dim (sparse, tile coding) | 2-dim (densa, normalizada) |
| Parâmetros aprendidos | ≈ 384 (lineares) | ≈ 4 800 (não-lineares) |
| Convergência típica | rápida (~50 ep) | mais lenta (~200 ep) |
| Estabilidade | muito estável | requer target net + clipping |
| Comprimento da política greedy | ~17 passos | ~17 passos (após convergir) |

**Discussão.** O DQN aprende *sem features manuais* — demonstra a abstração
end-to-end típica do deep RL. Em troca paga em **amostras necessárias** e
em **complexidade de treino** (replay buffer, target net, gradient clipping).
Para um problema deste tamanho, Linear SARSA com tile coding ganha
claramente em *sample efficiency*; o DQN brilha quando o espaço de estados
é demasiado grande para tile coding ser viável (continuous control,
imagens, etc.).

**Lições técnicas demonstradas:**
* O target network estabiliza o alvo TD evitando a "perseguição do próprio
  rabo" que apareceria se a mesma rede gerasse Q(s, a) e max Q(s', ·).
* O replay buffer descorrelaciona transições consecutivas — sem ele, o
  treino oscila.
* O épsilon decay (1.0 → 0.05) força exploração no início e
  explora-exploitation no final.

Reprodução: `python -m AR1.scripts.run_windy_gridworld_dqn --no-show`
(requer PyTorch). Implementação em `agents/control/dqn.py`, validada por
`tests/test_dqn.py` (8 testes, *skip* automático quando torch não está
instalado).

---

## Extensão Extra — AlphaZero-style no Tic-Tac-Toe

**Algoritmo:** reprodução em miniatura do AlphaZero (Silver et al., 2017/2018).
Não é uma reimplementação completa — é o algoritmo *do mesmo molde* aplicado a
um jogo solúvel para que se possam observar todas as peças a funcionar dentro
de um orçamento razoável.

**Componentes:**

* **Rede política+valor** — MLP único `27 → 64 → 64` com dois "heads":
  política `Linear(64, 9)` (9 logits, um por célula) e valor `Linear(64, 1) → tanh`
  com saída em `[-1, +1]`. Input: `encode_state(board, current_player)` (mesm