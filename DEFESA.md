# Notas para a Defesa Oral — Portefólio AR
**Data:** 09/06/2026 (online) · **Duração:** ~10 min + perguntas · **Aluno:** Luís Miguel Pereira Silva (PG60390)

---

## Estrutura sugerida (10 min)

| Tempo | Slide / Tópico |
|------:|----------------|
| 0:30 | **Abertura** — quem sou, o que cobre o portefólio (PL1–PL9), 5 ambientes, 18+ algoritmos |
| 1:30 | **Mapa da arquitetura** — pacote modular (envs / mdps / agents / experiments / scripts / tests) |
| 1:30 | **PL1–PL3 — Bandits + DP** — KArmed (UCB vs Thompson), Gridworld VI vs PI, Car Rental |
| 1:30 | **PL4–PL5 — Predição e Controlo** — Blackjack MC/TD/TD(n), Windy SARSA / Q-Learning / n-step |
| 1:30 | **PL6 — Aproximação Linear** — tile-coding 96-dim, NumPy vs PyTorch (mesma matemática) |
| 1:30 | **PL7–PL9 — Tic-Tac-Toe** — features 27-dim, SARSA/Q-Learn vs REINFORCE vs MCTS |
| 1:00 | **Engenharia** — 54 testes pytest, scripts reprodutíveis, `--no-show`, seeds |
| 1:00 | **Demo opcional** — `--play` para jogar contra MCTS (ou só mostrar a tabela final) |

---

## Mensagens-chave (uma por bloco)

* **Pacote modular:** trocar agente sem trocar ambiente, trocar experiência sem trocar agente.
* **DP:** Policy Iteration e Value Iteration convergem para a mesma V*; VI é mais rápido quando |A| é pequeno.
* **TD(n):** generaliza TD(0) (n=1) e MC (n=∞) — único corpo, dois extremos.
* **Tile-coding:** features lineares sem este truque não conseguem capturar o vento; com tile-coding capturam.
* **NumPy ≡ PyTorch:** `loss = ½(pred-target)² ⇒ grad = (pred-target)·φ` é exatamente `w += α·δ·φ`.
* **Q-Learning > SARSA no TicTacToe:** off-policy aprende a ótima diretamente, atinge 0 derrotas em 5000 jogos.
* **REINFORCE + baseline:** baseline V(s) reduz variância sem introduzir bias (78%→96% win rate).
* **MCTS:** 100% win rate vs Random com 1000 simulações *sem treino* — planeamento em tempo de decisão.

---

## Perguntas que podem fazer (e respostas curtas)

**P: Porque o ambiente Tic-Tac-Toe usa encoding relativo ao jogador (e não absoluto X/O)?**
R: Para que o mesmo agente possa jogar como X ou O sem retreinar — a perspetiva é "minhas peças / peças do adversário / vazio" em vez de "X / O / vazio". Confirmado por `test_encode_state_one_hot_perspective`.

**P: Como garante reprodutibilidade?**
R: Todos os scripts aceitam `--seed`. Os ambientes guardam `random.Random(seed)`. Os agentes (SARSA, Q-Learning, REINFORCE, MCTS) também. PyTorch usa `torch.manual_seed(0)` no `reset`.

**P: Diferença entre on-policy (SARSA) e off-policy (Q-Learning) na prática?**
R: No alvo TD: SARSA usa a *próxima ação realmente escolhida* (incluindo exploração); Q-Learning usa `max_a Q(s', a)` (a melhor possível). Resultado: Q-Learning aprende a política ótima ainda que explore; SARSA aprende uma política conservadora — ver tabela de derrotas no TicTacToe.

**P: Porque o MCTS bate o REINFORCE neste jogo?**
R: Tic-Tac-Toe tem horizonte muito curto (≤ 9 jogadas) — MCTS expande quase a árvore inteira com 1000 simulações. REINFORCE aprende uma *política amortizada* que aproxima a ótima mas perde para um agente que faz planeamento explícito por jogada.

**P: O que aconteceria se subisse n no n-step SARSA?**
R: n→∞ converge para MC Control (sem bootstrap, mais variância); n=1 é SARSA clássico (mais bias, menos variância). n=4 é um bom trade-off neste problema; em ambientes mais ruidosos ou episódios mais longos o n ótimo aumenta.

**P: Como sabe que a aproximação linear converge?**
R: Para semi-gradient TD(0)/SARSA linear, a teoria garante convergência sob condições de Robbins-Monro nas taxas de aprendizagem (α decrescente) e features limitadas. Empiricamente: os TD-errors decaem monotonicamente nas figuras `td_errors.png`.

**P: Porquê tile-coding em vez de uma rede neural?**
R: Para mostrar que o problema (capturar o efeito do vento) é resolúvel com aproximação **linear** se as features forem ricas o suficiente. Uma rede neural acrescentaria complexidade sem aprendizagem conceptual nova — o caminho NumPy↔PyTorch já demonstra a equivalência matemática.

**P: O que faria com mais tempo?**
R: Substituir Linear SARSA por DQN, treinar AlphaZero-style (MCTS + rede de política) no Tic-Tac-Toe, registar benchmarks em JSON para gráficos comparativos automáticos.

**P: O baseline V(s) no REINFORCE é treinado *à parte* ou em conjunto?**
R: Em conjunto, no mesmo passo MC: `delta = G_t − V(s_t)`, depois `theta += α·γ^t·delta·∇log π` e `w += α_w·delta·φ(s_t)`. É a versão da Sec. 13.4 do Sutton & Barto.

**P: O MCTS poderia usar uma rede neural em vez de rollout aleatório?**
R: Sim — seria o passo para AlphaZero. O rollout aleatório é o *default policy*; substituí-lo por uma rede treinada (value head + policy head) acelera drasticamente a convergência e é o que diferencia MCTS clássico de algoritmos como AlphaGo Zero.

---

## Demo possível (1-2 min)

```bash
# Mostrar terminal a abrir um jogo contra o MCTS
python -m AR1.scripts.run_mcts_tictactoe --play --play-sims 500

# Ou só correr os testes para mostrar que tudo passa:
PYTHONPATH=. pytest AR1/tests -q
```

---

## Checklist final antes da defesa

- [ ] Repositório pushed no GitHub (até 26/05 23h59)
- [ ] README e RESULTS.md atualizados
- [ ] `pytest AR1/tests` passa todos os 54 testes
- [ ] Outputs em `AR1/outputs/` gerados
- [ ] Conseguir explicar a função `_probs` do REINFORCE e o ciclo de 4 fases do MCTS sem olhar para o código
- [ ] Saber dizer o win rate de Q-Learning vs SARSA no Tic-Tac-Toe (98.7% vs 98.1%) e o motivo (off-policy)
- [ ] Ter o terminal pronto com `--play` para uma demo se for pedido

---

*Documento de apoio à defesa do portefólio individual de Aprendizagem por Reforço.*
