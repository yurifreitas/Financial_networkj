export type TendenciaMsg = {
  tempo: string;          // ISO
  preco: number;
  retorno_pred: number;   // ex.: 0.012 = +1.2%
  acao_modelo: -1 | 0 | 1;
  sinal_final: -1 | 0 | 1; // após estratégia
  posicao: -1 | 0 | 1;
};

export type MarkovNode = {
  t: number;          // passos à frente (horas, de acordo com backend)
  preco: number;
  ret: number;
  prob: number;       // 0..1
  data_futura: string;
};
