// Gerenciador de WebSocket com reconexão exponencial
export function wsConnect(path: string, onMessage: (ev: MessageEvent) => void) {
  const base = import.meta.env.VITE_BACKEND_WS || "ws://localhost:8000";
  let url = `${base}${path}`;

  let ws: WebSocket | null = null;
  let tries = 0;
  let timer: number | null = null;

  const connect = () => {
    ws = new WebSocket(url);
    ws.onopen = () => {
      tries = 0;
      console.info(`[WS] conectado: ${url}`);
    };
    ws.onmessage = onMessage;
    ws.onclose = () => {
      const delay = Math.min(15000, 500 * Math.pow(2, tries++));
      console.warn(`[WS] desconectado. tentando em ${delay}ms`);
      timer = window.setTimeout(connect, delay);
    };
    ws.onerror = (e) => {
      console.error("[WS] erro", e);
      ws?.close();
    };
  };

  connect();

  return () => {
    if (timer) window.clearTimeout(timer);
    ws?.close();
  };
}
