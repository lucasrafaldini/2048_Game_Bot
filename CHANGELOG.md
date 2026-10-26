# Changelog

All notable changes to this project will be documented in this file.

## [2.1.0] - 2026-10-26 - "Motor Bitboard de Alta Velocidade"

### ⚡ Engine rápido e autocontido (`game2048.core.fast`)

- **Bitboard 64-bit**: tabuleiro inteiro em um único inteiro (16 nibbles,
  um por célula, armazenando o expoente do tile: 0 = vazio, 1 = tile 2,
  ..., 15 = tile 32768).
- **Tabelas pré-computadas**: movimentos resolvem-se em 4 lookups de tabela
  (65536 entradas pré-calculadas para linha) - sem lógica de compressão
  em tempo de jogo.
- **Zero dependências**: apenas `random` + `dataclasses` da stdlib.
- **`FastGame`**: jogo completo autocontido, semeável (100% reprodutível),
  com log de spawns para replay determinístico em outros engines.
- **Policies rápidas**: `random_policy`, `greedy_policy`, `snake_policy`,
  `make_expectimax_policy(depth)` (expectimax no bitboard).
- **`batch_simulate`**: simulação em lote com `ProcessPoolExecutor`,
  seeds por jogo, totalmente reprodutível.
- **Throughput**: ~45.000 moves/s (núcleo único, policy aleatória).

### ✅ Acurácia verificada (`game2048.benchmarks.accuracy`)

- **Fuzzing de movimentos**: 20.000 comparações (5.000 tabuleiros aleatórios ×
  4 direções) contra o engine de referência `Board2048` - grid, score e flag
  de movimento idênticos. **0 divergências.**
- **Replay de jogos completos**: partida inteira do engine rápido é
  "replayada" turn-by-turn no engine de referência (spawns forçados
  idênticos) - grid e score idênticos após **cada** turno.
  17.243 turnos verificados, 0 divergências.
- **`AccuracyReport`**: contadores + pass/fail para CI.

### 📊 Estatística turn-by-turn

- `summarize_results`: média/mediana/dp/IC 95%, distribuição de max tile,
  win rate, throughput.
- `tile_reach_probabilities`: P(max ≥ 128/256/512/1024/2048).
- `turn_statistics`: curva de score por turno, melhor max tile por turno,
  curva de sobrevivência.

### 🖥️ CLI

- Novo comando `game2048-benchmark fast`: verifica acurácia contra o engine
  de referência, roda N jogos em paralelo e imprime estatística completa
  (summary + P(max tile) + evolução turn-by-turn). Exporta JSON com `-o`.

### 🧪 Qualidade

- 33 testes novos (`tests/test_fast.py`): equivalência vs referência,
  determinismo, distribuição de spawns (90/10), alinhamento do log de
  spawns, estatísticas, throughput. **92 testes no total, 100% pass.**
- Correção de bug no fast engine encontrada pelo próprio harness
  (`bits_to_grid` off-by-one) - prova de que o harness funciona.

## [2.0.0] - 2026-10-07 - "A Era das IAs & 10 Anos de Carreira"

### 🎯 Contexto
Esta versão marca uma transição simbólica: o código original (v1) foi escrito nos meus primeiros anos aprendendo a programar. Esta v2 representa 10 anos de evolução profissional e a chegada da era das IAs generativas.

---

### 🏷️ v1 - "Primeiros Passos" (Legado)
**Original:** https://github.com/lucasrafaldini/2048_Game_Bot

- Bot baseado em **screenshot + OCR de pixels** (PIL + PyAutoGUI)
- Coordenadas hardcoded da tela (`Cords.cord11` = (170, 270), etc.)
- Heurística simples com `scoreGrid` fixa
- Loop infinito `while True` jogando no navegador
- Zero testes, zero documentação técnica, zero estrutura
- **Objetivo:** Aprender Python, PIL, PyAutoGUI

> *"Esse repositório foi feito para que você possa programar o seu bot em Python sem muita dificuldade, com um material em português..."* — README original

---

### 🚀 v2 - "Engenharia Moderna" (Atual)

#### Core Engine
- **Pure Python** - Zero dependências externas para lógica do jogo
- **Board2048** - Tabuleiro imutável, serializável, testável
- **Game2048** - Orquestração com seeds reprodutíveis, histórico opcional
- **Direction enum** - Type-safe com vetores e oposites

#### IA Agents (6 algoritmos, 12 variantes)
| Categoria | Agentes |
|-----------|---------|
| Baseline | `random` |
| Heurísticas | `greedy`, `snake`, `monotonic` |
| Busca adversarial | `lookahead-2`, `lookahead-3` |
| **Expectimax** (padrão-ouro) | `expectimax-fast`, `expectimax`, `expectimax-deep` |
| **MCTS** | `mcts-fast`, `mcts`, `mcts-deep` |
| RL (opcional) | `DQN`, `PPO` via Gymnasium |

#### Benchmarking Profissional
- **CLI rico** (`game2048-benchmark`) com Typer + Rich
- **Execução paralela** com ProcessPoolExecutor
- **Configuração YAML** para experimentos reprodutíveis
- **Métricas completas**: score, max_tile, moves, tempo, nodes/ms

#### Data Science Stack
- **Estatística**: Mann-Whitney U, IC 95%, Cohen's d, Hedges' g
- **Armazenamento**: SQLite (metadados) + Parquet (big data)
- **Visualização**: Plotly (violin, stacked bars, scatter, animação)
- **Dashboard**: Streamlit interativo (`game2048-dashboard`)

#### Qualidade de Código
- **59 testes** (pytest) cobrindo core, agents, benchmarks, data
- **Type hints** completos + mypy strict
- **Ruff** para lint/format (substitui black/isort/flake8)
- **Pyproject.toml** moderno (PEP 621)
- **CI/CD ready** com GitHub Actions

#### Arquitetura Limpa
```
src/game2048/
├── core/       # Motor do jogo (sem deps)
├── ai/         # Agentes (pluggable)
├── benchmarks/ # Runner, métricas, CLI
├── visualization/ # Charts + Streamlit
├── data/       # Storage + Análise estatística
├── env.py      # Gymnasium env para RL
└── utils/      # Helpers puros
```

---

### 📊 Comparação v1 → v2

| Aspecto | v1 (201x) | v2 (2026) |
|---------|-----------|-----------|
| **Paradigma** | Script único | Package modular |
| **Detecção** | Screenshot + pixel OCR | Engine puro |
| **IA** | Heurística fixa | 6 algoritmos configuráveis |
| **Testes** | 0 | 59 (unit + integration) |
| **Benchmark** | Manual | Automatizado, paralelo, estatístico |
| **Dados** | Print no console | SQLite + Parquet + Dashboard |
| **Deploy** | `python main.py` | `pip install -e .` + CLI entrypoints |
| **Docs** | README tutorial | README + CHANGELOG + Type hints |

---

### 🎓 Lições da Jornada

1. **Código que funciona ≠ código mantível** - v1 jogava, v2 ensina e escala
2. **Abstração certa** - Separar engine de IA permite experimentação científica
3. **Reprodutibilidade** - Seeds, config YAML, armazenamento estruturado
4. **Métricas importam** - Não basta "ganhar", precisa medir *como* e *porquê*
5. **IA não substitui engenharia** - Expectimax vence, mas precisa de benchmarks rigorosos

---

### 🔮 Próximos Passos (v3?)

- [ ] Treinamento RL real (DQN/PPO com Stable-Baselines3)
- [ ] Otimização de expectimax com bitboards (C/Rust via PyO3)
- [ ] Interface web jogável (WebAssembly)
- [ ] Benchmark distribuído (Ray/Dask)
- [ ] Paper: "Comparative Analysis of 2048 Solvers"

---

**Tags Git:**
- `v1.0.0-legacy` - Código original (referência histórica)
- `v2.0.0` - Esta versão moderna