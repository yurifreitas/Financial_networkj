# ==============================================================
# 🧬 RuidoColepax — Ruído simbiótico quântico-fractal avançado
# ==============================================================
import torch
import math

class RuidoColepax:
    """
    Implementa ruídos simbióticos complexos e adaptativos inspirados em processos biológicos,
    quânticos e fractais, aplicáveis a tensores de ativação ou pesos de redes neurais.
    """

    def __init__(self,
                 base_intensity: float = 0.02,
                 anneal_rate: float = 0.000001,
                 fractal_layers: int = 3,
                 device: str = "cuda"):
        self.base_intensity = base_intensity
        self.anneal_rate = anneal_rate
        self.fractal_layers = fractal_layers
        self.device = device
        self.step = 0

    # ==========================================================
    # 🔹 Atualização simbiótica (chamada a cada iteração global)
    # ==========================================================
    def step_update(self):
        self.step += 1
        # decaimento simbiótico da intensidade
        self.current_intensity = max(
            0.002,
            self.base_intensity * math.exp(-self.step * self.anneal_rate)
        )

    # ==========================================================
    # 🌪 1. Ruído gaussiano adaptativo
    # ==========================================================
    def gaussian(self, x: torch.Tensor):
        sigma = self.current_intensity * (1 + 0.2 * math.sin(self.step / 2000.0))
        return x + torch.randn_like(x) * sigma

    # ==========================================================
    # 🌊 2. Ruído fractal multi-escala
    # ==========================================================
    def fractal(self, x: torch.Tensor):
        noise = torch.zeros_like(x)
        freq = 1.0
        amp = self.current_intensity
        for i in range(self.fractal_layers):
            noise += amp * torch.sin(freq * x + self.step * 0.002)
            noise += amp * torch.cos(freq * x * 1.37 + self.step * 0.001)
            freq *= 1.7
            amp *= 0.5
        return x + noise

    # ==========================================================
    # 🎚 3. Ruído pink (1/f)
    # ==========================================================
    def pink(self, x: torch.Tensor):
        shape = x.shape
        f = torch.fft.fftfreq(x.numel(), d=1.0, device=self.device).abs() + 1e-6
        spectrum = torch.randn_like(x.flatten()) / (f ** 0.5)
        pink_noise = torch.fft.ifft(torch.fft.fft(x.flatten()) + spectrum).real
        return x + pink_noise.view(shape) * self.current_intensity

    # ==========================================================
    # ⚛️ 4. Ruído quântico (log-normal)
    # ==========================================================
    def quantico(self, x: torch.Tensor):
        mu = 0.0
        sigma = self.current_intensity * 3
        qn = torch.exp(torch.randn_like(x) * sigma + mu) - 1.0
        return x + qn * torch.sign(x)

    # ==========================================================
    # 🌱 5. Ruído de dropout simbiótico (mutação sináptica)
    # ==========================================================
    def dropout_simbiotico(self, x: torch.Tensor, p: float = 0.02):
        mask = (torch.rand_like(x) > p).float()
        mutacao = torch.randn_like(x) * (p * self.current_intensity)
        return x * mask + mutacao * (1 - mask)

    # ==========================================================
    # 🌀 6. Ruído de campo topológico (harmônicos angulares)
    # ==========================================================
    def topologico(self, x: torch.Tensor):
        phase = torch.atan2(torch.sin(x * 2.1), torch.cos(x * 1.3))
        topo = torch.sin(phase * 3 + self.step * 0.003) * self.current_intensity * 2
        return x + topo

    # ==========================================================
    # 🧩 Aplicador genérico — mistura simbiótica de ruídos
    # ==========================================================
    def aplicar(self, x: torch.Tensor, modo: str = "mix"):
        self.step_update()

        if modo == "gauss":
            return self.gaussian(x)
        elif modo == "fractal":
            return self.fractal(x)
        elif modo == "pink":
            return self.pink(x)
        elif modo == "quantico":
            return self.quantico(x)
        elif modo == "drop":
            return self.dropout_simbiotico(x)
        elif modo == "topo":
            return self.topologico(x)
        elif modo == "mix":
            # Mistura simbiótica adaptativa
            x = self.fractal(x)
            x = self.gaussian(x)
            x = self.quantico(x)
            x = self.topologico(x)
            return x
        else:
            return x
