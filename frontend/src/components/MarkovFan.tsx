import React, { useMemo } from "react";
import Plot from "react-plotly.js";
import type { MarkovNode } from "../types";

// Desenha um “abanico” de cenários. Cada passo t gera múltiplos pontos
// com opacidade proporcional à probabilidade.

export default function MarkovFan({ nodes }: { nodes: MarkovNode[] }) {
  const groups = useMemo(() => {
    const byT = new Map<number, MarkovNode[]>();
    for (const n of nodes) {
      if (!byT.has(n.t)) byT.set(n.t, []);
      byT.get(n.t)!.push(n);
    }
    return [...byT.entries()].sort((a,b) => a[0]-b[0]);
  }, [nodes]);

  const traces: any[] = [];
  for (const [, arr] of groups) {
    const maxProb = Math.max(...arr.map(a => a.prob), 1e-9);
    traces.push({
      type: "scatter",
      mode: "markers",
      name: `t+${arr[0].t}`,
      x: arr.map(a => a.data_futura),
      y: arr.map(a => a.preco),
      marker: {
        size: arr.map(a => 8 + Math.sqrt(a.prob) * 10),
        opacity: arr.map(a => Math.max(0.1, a.prob / maxProb)),
        symbol: "circle-open"
      },
      hovertemplate: "t+%{text}h · %{x}<br>Preço: %{y:.2f}<br>Prob: %{customdata:.2%}<extra></extra>",
      text: arr.map(a => String(a.t)),
      customdata: arr.map(a => a.prob)
    });
  }

  const layout = {
    title: "Rede Markoviana — múltiplas previsões",
    paper_bgcolor: "#0f1523",
    plot_bgcolor: "#0f1523",
    font: { color: "#e7eaee" },
    margin: { l: 50, r: 20, t: 40, b: 40 },
    xaxis: { gridcolor: "#1c2742", showspikes: true, spikemode: "across" },
    yaxis: { gridcolor: "#1c2742", tickformat: ",.2f" },
    legend: { orientation: "h", y: -0.2 }
  };

  return (
    <Plot
      data={traces}
      layout={layout as any}
      useResizeHandler
      style={{ width: "100%", height: "420px" }}
      config={{ displayModeBar: true, responsive: true }}
    />
  );
}
