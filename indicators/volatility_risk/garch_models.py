from arch import arch_model
import numpy as np

def compute(close: np.ndarray, window=50, model_type='GARCH'):
    volatility_values = [None] * (window - 1)

    for i in range(window, len(close) + 1):
        subseries = close[i - window:i]

        # Reescalando os dados para melhor convergência (entre 1 e 1000)
        scale_factor = np.std(subseries)
        if scale_factor == 0:
            scale_factor = 1  # evitar divisão por zero

        scaled_series = subseries / scale_factor

        if model_type == 'GARCH':
            am = arch_model(scaled_series, vol='Garch', p=1, q=1, rescale=False)
        elif model_type == 'EGARCH':
            am = arch_model(scaled_series, vol='EGARCH', p=1, q=1, rescale=False)
        elif model_type == 'TGARCH':
            am = arch_model(scaled_series, vol='GARCH', p=1, o=1, q=1, rescale=False)
        else:
            raise ValueError("Model type not supported")

        try:
            res = am.fit(disp='off')
            forecast = res.forecast(horizon=1)

            # Retorna à escala original multiplicando pela escala ao quadrado (variância)
            variance = forecast.variance.values[-1, 0] * (scale_factor ** 2)
            volatility_values.append(float(variance))

        except Exception as e:
            print(f"Erro ao ajustar modelo {model_type} na janela {i}: {e}")
            volatility_values.append(None)

    return volatility_values