// =========================================
// 🤖 RobotStatus.tsx — versão expandida
// =========================================
import React from "react";

export default function RobotStatus({ tick }: { tick: any }) {
  const corAcao =
    tick.acao === "BUY"
      ? "text-emerald-400"
      : tick.acao === "SELL"
      ? "text-rose-400"
      : "text-gray-400";

  const emPosicao = tick.posicao > 0;
  const pnl = emPosicao
    ? ((tick.preco - (tick.preco_entrada || tick.preco)) /
        (tick.preco_entrada || tick.preco)) *
      100
    : 0;

  const tendencia =
    tick.retorno_pred > 0.002
      ? { label: "📈 Alta", cor: "text-emerald-400" }
      : tick.retorno_pred < -0.002
      ? { label: "📉 Queda", cor: "text-rose-400" }
      : { label: "⚖️ Neutro", cor: "text-gray-300" };

  // 🧩 cálculos adicionais
  const probBaixa = 1 - (tick.prob_alta || 0);
  const forcaPrev = Math.abs(tick.retorno_pred || 0) * 100;
  const tempoUltima =
    tick.tempo_execucao &&
    `${Math.round((Date.now() - new Date(tick.tempo_execucao).getTime()) / 1000)}s`;

  const variacaoPreco =
    tick.preco_entrada && tick.preco_entrada > 0
      ? ((tick.preco - tick.preco_entrada) / tick.preco_entrada) * 100
      : 0;

  const coerencia =
    tick.coerencia || (1 - Math.abs(tick.retorno_pred)) * (tick.energia || 1);

  return (
    <div className="relative overflow-hidden p-6 bg-gradient-to-br from-gray-900/80 via-gray-800/60 to-gray-900/80 rounded-2xl border border-gray-800 shadow-xl backdrop-blur-md transition-all hover:shadow-cyan-900/20">
      {/* Fundo simbiótico */}
      <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/10 via-transparent to-indigo-500/10 animate-pulse blur-3xl pointer-events-none" />

      <div className="relative z-10">
        {/* Cabeçalho */}
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-2xl font-extrabold flex items-center gap-2">
            🤖 Estado do Robô
            <span className="text-sm text-gray-400 font-medium">
              {new Date(tick.tempo).toLocaleTimeString()}
            </span>
          </h2>

          <p
            className={`text-2xl font-bold ${corAcao} transition-transform duration-500 ${
              tick.acao !== "-" ? "animate-pulse" : ""
            }`}
          >
            {tick.acao === "-" ? "Aguardando..." : tick.acao}
          </p>
        </div>

        {/* Grid de informações */}
        <div className="grid grid-cols-2 md:grid-cols-3 gap-5 text-sm">
          {/* === Linha 1 === */}
          <div>
            <p className="text-gray-400">💲 Preço Atual</p>
            <p className="text-lg font-semibold">${tick.preco.toFixed(2)}</p>
          </div>

          <div>
            <p className="text-gray-400">🎯 Preço de Entrada</p>
            <p className="text-lg font-semibold">
              {tick.preco_entrada ? `$${tick.preco_entrada.toFixed(2)}` : "—"}
            </p>
          </div>

          <div>
            <p className="text-gray-400">📦 Posição</p>
            <p className="text-lg font-semibold">
              {tick.posicao > 0
                ? `${tick.posicao.toFixed(5)} BTC`
                : "Nenhuma posição"}
            </p>
          </div>

          {/* === Linha 2 === */}
          <div>
            <p className="text-gray-400">💰 Capital Livre</p>
            <p className="text-lg font-semibold">${tick.capital.toFixed(2)}</p>
          </div>

          <div>
            <p className="text-gray-400">🏦 Patrimônio Total</p>
            <p className="text-lg font-semibold">${tick.patrimonio.toFixed(2)}</p>
          </div>

          <div>
            <p className="text-gray-400">📊 Lucro Atual</p>
            <p
              className={`text-lg font-semibold ${
                pnl > 0
                  ? "text-emerald-400"
                  : pnl < 0
                  ? "text-rose-400"
                  : "text-gray-300"
              }`}
            >
              {pnl.toFixed(2)}%
            </p>
          </div>

          {/* === Linha 3 === */}
          <div>
            <p className="text-gray-400">📈 Retorno Acumulado</p>
            <p
              className={`text-lg font-semibold ${
                tick.retorno_pct > 0
                  ? "text-emerald-400"
                  : tick.retorno_pct < 0
                  ? "text-rose-400"
                  : "text-gray-300"
              }`}
            >
              {tick.retorno_pct.toFixed(2)}%
            </p>
          </div>

          <div>
            <p className="text-gray-400">🔮 Tendência Prevista</p>
            <p className={`text-lg font-semibold ${tendencia.cor}`}>
              {tendencia.label}
            </p>
          </div>

          <div>
            <p className="text-gray-400">⚡ Energia Simbiótica</p>
            <p
              className="text-lg font-semibold text-cyan-400 animate-pulse"
              title="Força simbiótica do modelo (coerência preditiva)"
            >
              {(tick.energia * 100).toFixed(1)}%
            </p>
          </div>

          {/* === Linha 4 — novos indicadores === */}
          <div>
            <p className="text-gray-400">🧠 Prob. Alta</p>
            <p className="text-lg font-semibold text-emerald-400">
              {(tick.prob_alta * 100).toFixed(2)}%
            </p>
          </div>

          <div>
            <p className="text-gray-400">⚠️ Prob. Baixa</p>
            <p className="text-lg font-semibold text-rose-400">
              {(probBaixa * 100).toFixed(2)}%
            </p>
          </div>

          <div>
            <p className="text-gray-400">📡 Força da Previsão</p>
            <p
              className={`text-lg font-semibold ${
                forcaPrev > 0.5 ? "text-cyan-300" : "text-gray-400"
              }`}
            >
              {forcaPrev.toFixed(2)}%
            </p>
          </div>

          <div>
            <p className="text-gray-400">⏱ Última Execução</p>
            <p className="text-lg font-semibold text-gray-300">
              {tempoUltima || "—"}
            </p>
          </div>

          <div>
            <p className="text-gray-400">📉 Variação Desde Entrada</p>
            <p
              className={`text-lg font-semibold ${
                variacaoPreco > 0
                  ? "text-emerald-400"
                  : variacaoPreco < 0
                  ? "text-rose-400"
                  : "text-gray-400"
              }`}
            >
              {variacaoPreco.toFixed(2)}%
            </p>
          </div>

          <div>
            <p className="text-gray-400">🪶 Coerência Simbiótica</p>
            <p className="text-lg font-semibold text-indigo-300">
              {(coerencia * 100).toFixed(1)}%
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
