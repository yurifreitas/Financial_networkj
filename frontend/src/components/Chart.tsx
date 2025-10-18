import Plot from "react-plotly.js";

export default function Chart({ dados }: { dados: any[] }) {
  if (!dados || dados.length === 0) return <p>Carregando gráfico...</p>;

  // pontos de compra e venda
  const buys = dados.filter((d) => d.acao === "BUY");
  const sells = dados.filter((d) => d.acao === "SELL");

  // limites de eixos
  const minPreco = Math.min(...dados.map((d) => d.preco));
  const maxPreco = Math.max(...dados.map((d) => d.preco));
  const minPat = Math.min(...dados.map((d) => d.patrimonio));
  const maxPat = Math.max(...dados.map((d) => d.patrimonio));

  // linha da tendência prevista
  const tendencia = dados.map((d) =>
    d.retorno_pred > 0.002 ? 1 : d.retorno_pred < -0.002 ? -1 : 0
  );

  return (
    <div className="bg-gray-900 rounded-xl p-4 mt-4 shadow-lg">
      <h2 className="text-xl font-semibold text-white mb-2">
        💹 Evolução de Preço, Patrimônio e Operações
      </h2>
      <Plot
        data={[
          // 🔷 Preço principal
          {
            x: dados.map((d) => d.tempo),
            y: dados.map((d) => d.preco),
            type: "scatter",
            mode: "lines",
            name: "Preço BTC/USDT",
            line: { color: "#ddd", width: 2 },
            yaxis: "y1",
          },
          // 💰 Patrimônio
          {
            x: dados.map((d) => d.tempo),
            y: dados.map((d) => d.patrimonio),
            type: "scatter",
            mode: "lines",
            name: "Patrimônio (USD)",
            line: { color: "cyan", width: 2, dash: "dot" },
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
              color: "#00ff7f",
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
              color: "#ff4040",
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
            name: "Tendência prevista",
            line: { color: "orange", width: 1.5 },
            yaxis: "y3",
          },
        ]}
        layout={{
          paper_bgcolor: "#111",
          plot_bgcolor: "#111",
          font: { color: "white" },
          height: 600,
          margin: { l: 70, r: 70, t: 40, b: 60 },
          xaxis: {
            title: "Tempo",
            showgrid: false,
            tickformat: "%H:%M",
          },
          yaxis: {
            title: "Preço BTC/USDT",
            color: "#ddd",
            range: [minPreco * 0.998, maxPreco * 1.002],
            domain: [0.25, 1],
          },
          yaxis2: {
            title: "Patrimônio (USD)",
            overlaying: "y",
            side: "right",
            showgrid: false,
            color: "cyan",
            range: [minPat * 0.998, maxPat * 1.002],
          },
          yaxis3: {
            title: "Tendência (Rede)",
            side: "left",
            anchor: "free",
            overlaying: "y",
            position: 0.05,
            showgrid: false,
            range: [-1.5, 1.5],
            tickvals: [-1, 0, 1],
            ticktext: ["📉", "⚖️", "📈"],
          },
          legend: {
            orientation: "h",
            x: 0.5,
            xanchor: "center",
            y: -0.15,
          },
        }}
        config={{
          displayModeBar: true,
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
