import React, { useMemo } from "react";
import Plot from "react-plotly.js";

type Point = { t: string; y: number };
type Trade = { t: string; y: number; side: "BUY" | "SELL" | "EXIT" };

export default function PriceChart({
  series,
  trades
}: {
  series: Point[];
  trades: Trade[];
}) {
  const x = series.map(p => p.t);
  const y = series.map(p => p.y);

  const buyPts = trades.filter(t => t.side === "BUY");
  const sellPts = trades.filter(t => t.side === "SELL");
  const exitPts = trades.filter(t => t.side === "EXIT");

  const layout = useMemo(() => ({
    paper_bgcolor: "#0f1523",
    plot_bgcolor: "#0f1523",
    font: { color: "#e7eaee" },
    margin: { l: 50, r: 20, t: 20, b: 30 },
    xaxis: { title: "", gridcolor: "#1c2742", showspikes: true, spikemode: "across" },
    yaxis: { title: "Preço", gridcolor: "#1c2742", tickformat: ",.2f" },
    legend: { orientation: "h", y: -0.2 },
    dragmode: "pan",
    hovermode: "x unified",
    shapes: [] as any[]
  }), []);

  const data: any[] = [
    {
      type: "scatter",
      mode: "lines",
      name: "Preço",
      x,
      y,
      line: { width: 2 }
    },
    {
      type: "scatter",
      mode: "markers",
      name: "BUY",
      x: buyPts.map(p => p.t),
      y: buyPts.map(p => p.y),
      marker: { size: 10, symbol: "triangle-up", line: { width: 1 } }
    },
    {
      type: "scatter",
      mode: "markers",
      name: "SELL",
      x: sellPts.map(p => p.t),
      y: sellPts.map(p => p.y),
      marker: { size: 10, symbol: "triangle-down", line: { width: 1 } }
    },
    {
      type: "scatter",
      mode: "markers",
      name: "EXIT",
      x: exitPts.map(p => p.t),
      y: exitPts.map(p => p.y),
      marker: { size: 9, symbol: "x", line: { width: 1 } }
    }
  ];

  return (
    <Plot
      data={data}
      layout={layout as any}
      useResizeHandler
      style={{ width: "100%", height: "520px" }}
      config={{ displayModeBar: true, responsive: true }}
    />
  );
}
