import Plot from "react-plotly.js";

export default function Chart({ dados }: { dados: any[] }) {
  if (!dados || dados.length === 0)
    return <p className="text-gray-400 text-center mt-4">Carregando gráfico...</p>;

  // Filtra pontos de trade
  const buys = dados.filter((d) => d.acao === "BUY");
  const sells = dados.filter((d) => d.acao === "SELL");

  // Limites
  const minPreco = Math.min(...dados.map((d) => d.preco));
  const maxPreco = Math.max(...dados.map((d) => d.preco));
  const minPat = Math.min(...dados.map((d) => d.patrimonio));
  const maxPat = Math.max(...dados.map((d) => d.patrimonio));

  const tendencia = dados.map((d) =>
    d.retorno_pred > 0.002 ? 1 : d.retorno_pred < -0.002 ? -1 : 0
  );

  return (
    <div className="bg-gray-900/90 rounded-2xl p-5 mt-6 shadow-lg border border-gray-800">
      <h2 className="text-2xl font-semibold text-white mb-3 flex items-center gap-2">
        💹 Visualização Simbiótica — Preço x Previsão x Posição
      </h2>

      <Plot
        data={[
          // 🩵 Área entre previsão e preço real (erro simbiótico)
          {
            x: dados.map((d) => d.tempo),
            y: dados.map((d) => d.preco),
            type: "scatter",
            mode: "lines",
            fill: "tonexty",
            fillcolor: "rgba(0, 255, 255, 0.1)",
            line: { color: "transparent" },
            showlegend: false,
            yaxis: "y1",
          },
          {
            x: dados.map((d) => d.tempo),
            y: dados.map((d) => d.preco_previsto ?? d.preco),
            type: "scatter",
            mode: "lines",
            fill: "tonexty",
            fillcolor: "rgba(0, 255, 255, 0.1)",
            line: { color: "transparent" },
            showlegend: false,
            yaxis: "y1",
          },
          // ⚪ Preço real
          {
            x: dados.map((d) => d.tempo),
            y: dados.map((d) => d.preco),
            type: "scatter",
            mode: "lines",
            name: "Preço Real",
            line: { color: "#f8fafc", width: 2 },
            yaxis: "y1",
          },
          // 🔮 Preço previsto
          {
            x: dados.map((d) => d.tempo),
            y: dados.map((d) => d.preco_previsto ?? d.preco),
            type: "scatter",
            mode: "lines",
            name: "Preço Previsto (Rede Markoviana)",
            line: { color: "#00ffff", width: 2, dash: "dot" },
            yaxis: "y1",
          },
          // 💰 Patrimônio
          {
            x: dados.map((d) => d.tempo),
            y: dados.map((d) => d.patrimonio),
            type: "scatter",
            mode: "lines",
            name: "Patrimônio (USD)",
            line: { color: "#22d3ee", width: 2 },
            yaxis: "y2",
          },
          // 🟢 Entradas
          {
            x: buys.map((b) => b.tempo),
            y: buys.map((b) => b.preco),
            type: "scatter",
            mode: "markers",
            name: "Compra",
            marker: {
              color: "#22ff88",
              size: 10,
              symbol: "triangle-up",
              line: { color: "black", width: 1 },
            },
            yaxis: "y1",
          },
          // 🔴 Saídas
          {
            x: sells.map((s) => s.tempo),
            y: sells.map((s) => s.preco),
            type: "scatter",
            mode: "markers",
            name: "Venda",
            marker: {
              color: "#ff4444",
              size: 10,
              symbol: "triangle-down",
              line: { color: "black", width: 1 },
            },
            yaxis: "y1",
          },
          // ⚖️ Tendência simbiótica
          {
            x: dados.map((d) => d.tempo),
            y: tendencia,
            type: "scatter",
            mode: "lines",
            name: "Tendência (Rede)",
            line: { color: "#fbbf24", width: 1.5 },
            yaxis: "y3",
          },
          // 🎯 Preço de entrada ativo (destaque único)
          ...dados
            .filter((d) => d.preco_entrada)
            .map((d, i) => ({
              x: [d.tempo],
              y: [d.preco_entrada],
              type: "scatter",
              mode: "markers+text",
              text: ["Entrada"],
              textposition: "top center",
              marker: {
                color: "#fde047",
                size: 9,
                symbol: "star",
                line: { color: "#facc15", width: 1 },
              },
              name: `Entrada #${i + 1}`,
              yaxis: "y1",
              showlegend: i === 0, // mostra uma vez só na legenda
            })),
        ]}
        layout={{
          paper_bgcolor: "#0f172a",
          plot_bgcolor: "#0f172a",
          font: { color: "#e2e8f0" },
          height: 640,
          margin: { l: 70, r: 70, t: 40, b: 60 },
          xaxis: {
            title: "Tempo (UTC)",
            showgrid: false,
            tickformat: "%H:%M",
          },
          yaxis: {
            title: "Preço BTC/USDT",
            color: "#94a3b8",
            range: [minPreco * 0.998, maxPreco * 1.002],
            domain: [0.25, 1],
          },
          yaxis2: {
            title: "Patrimônio (USD)",
            overlaying: "y",
            side: "right",
            color: "#22d3ee",
            range: [minPat * 0.998, maxPat * 1.002],
          },
          yaxis3: {
            title: "Tendência Rede",
            side: "left",
            anchor: "free",
            overlaying: "y",
            position: 0.05,
            range: [-1.5, 1.5],
            tickvals: [-1, 0, 1],
            ticktext: ["📉", "⚖️", "📈"],
            color: "#fbbf24",
          },
          legend: {
            orientation: "h",
            x: 0.5,
            xanchor: "center",
            y: -0.18,
          },
        }}
        config={{
          responsive: true,
          scrollZoom: true,
          displaylogo: false,
          modeBarButtonsToRemove: [
            "toImage",
            "zoom2d",
            "pan2d",
            "select2d",
            "lasso2d",
          ],
        }}
        style={{ width: "100%" }}
      />
    </div>
  );
}
