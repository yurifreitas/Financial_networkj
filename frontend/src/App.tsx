import { useEffect, useState } from "react";
import Chart from "./components/Chart";
import RobotStatus from "./components/RobotStatus";
import TradeHistory from "./components/TradeHistory";
import MarkovForecast from "./components/MarkovForecast";

// ==========================
// 🧠 Tipagem simbiótica do Tick
// ==========================
interface Tick {
  tempo: string;
  preco: number;
  preco_previsto?: number;
  preco_entrada?: number | null;
  acao: string;
  capital: number;
  posicao: number;
  patrimonio: number;
  retorno_pct: number;
  retorno_pred?: number;
  prob_alta?: number;
  energia?: number;
}

// ==========================
// 🌌 Componente principal
// ==========================
export default function App() {
  const [data, setData] = useState<Tick[]>([]);
  const [ultimo, setUltimo] = useState<Tick | null>(null);
  const [status, setStatus] = useState<"conectando" | "ativo" | "offline">("conectando");

  useEffect(() => {
    const ws = new WebSocket("ws://localhost:8003/ws/robo");

    ws.onopen = () => setStatus("ativo");
    ws.onclose = () => setStatus("offline");
    ws.onerror = () => setStatus("offline");

    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        setData((prev) => [...prev.slice(-300), msg]);
        setUltimo(msg);
      } catch (e) {
        console.error("Erro ao processar mensagem do robô:", e);
      }
    };

    return () => ws.close();
  }, []);

  // ==========================
  // 🎨 UI simbiótica com Tailwind
  // ==========================
  return (
    <div className="bg-gradient-to-br from-gray-900 via-black to-gray-950 text-gray-100 min-h-screen flex flex-col">
      {/* ====== Cabeçalho ====== */}
      <header className="border-b border-gray-800 px-6 py-4 flex items-center justify-between backdrop-blur-md bg-white/5">
        <h1 className="text-3xl font-extrabold tracking-tight text-cyan-400 flex items-center gap-2">
          <span>🤖</span> EtherSym Finance — Live Trader
        </h1>

        <div
          className={`px-3 py-1 rounded-full text-sm font-semibold transition-colors duration-300
          ${
            status === "ativo"
              ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/40"
              : status === "conectando"
              ? "bg-yellow-500/20 text-yellow-400 border border-yellow-500/40"
              : "bg-red-500/20 text-red-400 border border-red-500/40"
          }`}
        >
          {status === "ativo"
            ? "Conectado"
            : status === "conectando"
            ? "Conectando..."
            : "Desconectado"}
        </div>
      </header>

      {/* ====== Conteúdo Principal ====== */}
      <main className="flex-1 container mx-auto px-6 py-8 grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Painel lateral de status e histórico */}
        <div className="lg:col-span-1 space-y-6">
          {ultimo && (
            <div className="bg-white/5 rounded-2xl p-6 shadow-xl border border-gray-800 backdrop-blur-md hover:bg-white/10 transition-all">
              <RobotStatus tick={ultimo} />
            </div>
          )}

          <div className="bg-white/5 rounded-2xl p-4 shadow-xl border border-gray-800 backdrop-blur-md overflow-hidden">
            <h2 className="text-xl font-semibold mb-3 text-cyan-300">
              Histórico de Operações
            </h2>
            <TradeHistory dados={data} />
          </div>
        </div>

        {/* Gráfico principal + previsões */}
        <div className="lg:col-span-2 flex flex-col gap-6">
          <div className="bg-white/5 rounded-2xl p-6 shadow-xl border border-gray-800 backdrop-blur-md hover:bg-white/10 transition-all">
            <h2 className="text-xl font-semibold mb-3 text-cyan-300">
              Evolução Patrimonial
            </h2>
            <Chart dados={data} />
          </div>

        
        </div>
      </main>

      {/* ====== Rodapé ====== */}
      <footer className="border-t border-gray-800 text-center text-sm py-4 text-gray-500">
        <span className="text-gray-400">🌌 EtherSym Finance</span> — modo simbiótico ativo ·{" "}
        <span className="text-cyan-400 font-semibold">Realtime Preditivo</span>
      </footer>
    </div>
  );
}
