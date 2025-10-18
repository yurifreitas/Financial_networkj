import { useEffect, useState } from "react";
import Chart from "./components/Chart";
import RobotStatus from "./components/RobotStatus";
import TradeHistory from "./components/TradeHistory";

interface Tick {
  tempo: string;
  preco: number;
  acao: string;
  capital: number;
  posicao: number;
  patrimonio: number;
  retorno_pct: number;
}

export default function App() {
  const [data, setData] = useState<Tick[]>([]);
  const [ultimo, setUltimo] = useState<Tick | null>(null);

  useEffect(() => {
    const ws = new WebSocket("ws://localhost:8003/ws/robo");
    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      setData((prev) => [...prev.slice(-200), msg]);
      setUltimo(msg);
    };
    return () => ws.close();
  }, []);

  return (
    <div className="bg-gray-900 text-white min-h-screen p-4">
      <h1 className="text-3xl font-bold mb-4">🤖 EtherSym Trader Live</h1>
      {ultimo && <RobotStatus tick={ultimo} />}
      <Chart dados={data} />
      <TradeHistory dados={data} />
    </div>
  );
}
