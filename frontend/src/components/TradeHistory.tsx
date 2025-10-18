export default function TradeHistory({ dados }: { dados: any[] }) {
  const trades = dados.filter((d) => d.acao !== "-").slice(-10).reverse();
  return (
    <div className="mt-6 bg-gray-800 p-3 rounded-lg">
      <h2 className="text-lg font-bold mb-2">📜 Últimas operações</h2>
      <ul>
        {trades.map((t, i) => (
          <li key={i} className="border-b border-gray-700 py-1">
            <span className="text-sm text-gray-400">{new Date(t.tempo).toLocaleString()} — </span>
            <span className={t.acao === "BUY" ? "text-green-400" : "text-red-400"}>
              {t.acao}
            </span>{" "}
            @ ${t.preco.toFixed(2)} | Patrimônio: ${t.patrimonio.toFixed(2)}
          </li>
        ))}
      </ul>
    </div>
  );
}
