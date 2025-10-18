// =========================================================
// 📈 MarkovForecast.tsx — Visualização Rede Markoviana
// =========================================================
import { useEffect, useState } from "react";
import Plot from "react-plotly.js";

interface MarkovData {
  data_futura: string;
  preco: number;
  cenario: "otimista" | "pessimista" | "neutro";
  preco_medio: number;
}

export default function MarkovForecast() {
  const [dados, setDados] = useState<MarkovData[]>([]);
  const [ultimoPreco, setUltimoPreco] = useState<number | null>(null);
  const [status, setStatus] = useState("Conectando...");

  useEffect(() => {
    const ws = new WebSocket("ws://localhost:8003/ws/markov");
    ws.onopen = () => setStatus("🧠 Conectado à Rede Markoviana");
    ws.onclose = () => setStatus("🔴 Desconectado");
    ws.onerror = () => setStatus("⚠️ Erro de conexão");

    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        if (Array.isArray(msg)) setDados(msg);
      } catch (err) {
        console.error("Erro ao decodificar Markov:", err);
      }
    };

    return () => ws.close();
  }, []);

  // separa os cenários
  const otimista = dados.filter((d) => d.cenario === "otimista");
  const neutro = dados.filter((d) => d.cenario === "neutro");
  const pessimista = dados.filter((d) => d.cenario === "pessimista");

  // preço médio ponderado (valor central da previsão)
  const media = dados.filter(
    (v, i, arr) =>
      arr.findIndex((x) => x.data_futura === v.data_futura) === i
  );

  // garante escalas coerentes
  const maxPreco = Math.max(...dados.map((d) => d.preco), 0);
  const minPreco = Math.min(...dados.map((d) => d.preco), maxPreco * 0.9);

  return (
    <div className="bg-white/5 rounded-2xl p-6 shadow-xl border border-gray-800 backdrop-blur-md mt-6">
      <h2 className="text-xl font-semibold text-cyan-300 mb-2">
        Previsões Futuras (Rede Markoviana)
      </h2>
      <p className="text-sm text-gray-400 mb-4">{status}</p>

      {dados.length === 0 ? (
        <p className="text-gray-500">Aguardando previsões...</p>
      ) : (
        <Plot
          data={[
            {
              x: otimista.map((d) => d.data_futura),
              y: otimista.map((d) => d.preco),
              type: "scatter",
              mode: "lines+markers",
              name: "Cenário Otimista",
              line: { color: "limegreen", width: 2 },
            },
            {
              x: neutro.map((d) => d.data_futura),
              y: neutro.map((d) => d.preco),
              type: "scatter",
              mode: "lines",
              name: "Cenário Neutro",
              line: { color: "deepskyblue", width: 1.5, dash: "dot" },
            },
            {
              x: pessimista.map((d) => d.data_futura),
              y: pessimista.map((d) => d.preco),
              type: "scatter",
              mode: "lines",
              name: "Cenário Pessimista",
              line: { color: "tomato", width: 2 },
            },
            {
              x: media.map((d) => d.data_futura),
              y: media.map((d) => d.preco_medio),
              type: "scatter",
              mode: "lines",
              name: "Preço Médio Ponderado",
              line: { color: "#00ffff", width: 2, dash: "dot" },
            },
          ]}
          layout={{
            paper_bgcolor: "#0b0f19",
            plot_bgcolor: "#0b0f19",
            font: { color: "white" },
            height: 480,
            margin: { l: 70, r: 40, t: 50, b: 50 },
            xaxis: {
              title: "Tempo Futuro (UTC)",
              showgrid: false,
              tickformat: "%H:%M",
            },
            yaxis: {
              title: "Preço BTC/USDT",
              range: [minPreco * 0.999, maxPreco * 1.001],
              color: "#ccc",
            },
            legend: {
              orientation: "h",
              x: 0.5,
              xanchor: "center",
              y: -0.2,
              font: { size: 11 },
            },
          }}
          config={{
            displayModeBar: false,
            responsive: true,
            scrollZoom: true,
          }}
          style={{ width: "100%" }}
        />
      )}
    </div>
  );
}
