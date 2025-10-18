export default function RobotStatus({ tick }: { tick: any }) {
  const corAcao =
    tick.acao === "BUY"
      ? "text-green-400"
      : tick.acao === "SELL"
      ? "text-red-400"
      : "text-gray-400";

  const emPosicao = tick.posicao > 0;
  const pnl = emPosicao
    ? ((tick.preco - (tick.preco_entrada || tick.preco)) /
        (tick.preco_entrada || tick.preco)) *
      100
    : 0;

  const tendencia =
    tick.retorno_pred > 0.002
      ? "📈 Alta"
      : tick.retorno_pred < -0.002
      ? "📉 Queda"
      : "⚖️ Neutro";

  return (
    <div className="p-5 bg-gray-800 rounded-xl shadow-md mb-5 border border-gray-700">
      <div className="flex justify-between items-center mb-3">
        <h2 className="text-2xl font-bold">
          🤖 Estado do Robô{" "}
          <span className="text-gray-400 text-sm ml-2">
            ({new Date(tick.tempo).toLocaleTimeString()})
          </span>
        </h2>
        <p className={`text-xl font-semibold ${corAcao}`}>
          {tick.acao === "-" ? "Aguardando sinal..." : tick.acao}
        </p>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-3 gap-3 text-sm">
        <div>
          <p className="text-gray-400">Preço Atual</p>
          <p className="text-lg font-semibold">${tick.preco.toFixed(2)}</p>
        </div>

        <div>
          <p className="text-gray-400">Preço de Entrada</p>
          <p className="text-lg font-semibold">
            {tick.preco_entrada ? `$${tick.preco_entrada.toFixed(2)}` : "—"}
          </p>
        </div>

        <div>
          <p className="text-gray-400">Posição</p>
          <p className="text-lg font-semibold">
            {tick.posicao > 0
              ? `${tick.posicao.toFixed(5)} BTC`
              : "Nenhuma posição"}
          </p>
        </div>

        <div>
          <p className="text-gray-400">Capital Livre</p>
          <p className="text-lg font-semibold">${tick.capital.toFixed(2)}</p>
        </div>

        <div>
          <p className="text-gray-400">Patrimônio Total</p>
          <p className="text-lg font-semibold">${tick.patrimonio.toFixed(2)}</p>
        </div>

        <div>
          <p className="text-gray-400">Lucro Operação Atual</p>
          <p
            className={`text-lg font-semibold ${
              pnl > 0 ? "text-green-400" : pnl < 0 ? "text-red-400" : "text-gray-300"
            }`}
          >
            {pnl.toFixed(2)}%
          </p>
        </div>

        <div>
          <p className="text-gray-400">Retorno Acumulado</p>
          <p
            className={`text-lg font-semibold ${
              tick.retorno_pct > 0
                ? "text-green-400"
                : tick.retorno_pct < 0
                ? "text-red-400"
                : "text-gray-300"
            }`}
          >
            {tick.retorno_pct.toFixed(2)}%
          </p>
        </div>

        <div>
          <p className="text-gray-400">Tendência Prevista</p>
          <p className="text-lg font-semibold">{tendencia}</p>
        </div>

        {tick.energia && (
          <div>
            <p className="text-gray-400">Energia Simbiótica</p>
            <p className="text-lg font-semibold">
              {(tick.energia * 100).toFixed(1)}%
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
