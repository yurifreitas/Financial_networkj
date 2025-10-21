# 🌌 EtherSym Finance
🌱 EtherSym Finance é o resultado de uma longa jornada em busca de traduzir princípios biológicos em comportamento computacional.
Ao longo dos anos, venho me aprofundando em interfaces simbióticas — sistemas capazes de aprender, se regular e evoluir, inspirados em processos vivos como homeostase, regeneração e equilíbrio energético.

O projeto nasce do desejo de aproximar a biologia da inteligência artificial: construir redes que não apenas executem tarefas, mas vivam dentro de seus próprios parâmetros, adaptando-se em função da energia, do erro e da experiência.

Tecnicamente, o EtherSym Finance utiliza aprendizado por reforço profundo (Deep Reinforcement Learning) com uma arquitetura Dueling DQN estendida, combinando:
---

## 🧠 Estrutura

- **Rede Dueling DQN + Cabeça de Regressão Contínua**
  - Ações discretas (comprar / segurar / vender)
  - Saída contínua de previsão de retorno futuro
- **Mecanismos simbióticos**
  - Homeostase adaptativa
  - Poda e regeneração sináptica
  - Replay buffer com priorização
- **Treinamento**
  - Dados de candles históricos (1h) — Binance API
  - Suporte a GPU com PyTorch e `torch.compile`
  - Estados persistentes e métricas Sharpe-like

---
Run Project
```bash
python -m v1.main
```