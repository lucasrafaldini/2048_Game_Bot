# Changelog

All notable changes to this project will be documented in this file.

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