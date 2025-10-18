# ==========================================
# 🧠 EtherSym Finance — Rede Dueling DQN + Regressão de Preço
# ==========================================
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from config import LR


class RedeAvancada(nn.Module):
    def __init__(self, state_dim=10, n_actions=3):
        super().__init__()

        # === Backbone simbiótico ===
        self.fc1 = nn.Linear(state_dim, 128)
        self.norm1 = nn.LayerNorm(128)
        self.fc2 = nn.Linear(128, 64)
        self.norm2 = nn.LayerNorm(64)
        self.skip = nn.Linear(128, 64)
        self.act = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(0.1)

        # === Cabeças ===
        # Valor e vantagem (dueling)
        self.val = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
        self.adv = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, n_actions))

        # Nova cabeça: regressão de preço (retorno futuro contínuo)
        self.reg = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))

        # Buffers simbióticos
        self.historico = []
        self.media_antiga = None
        self.estavel = 0
        self.limiar_homeostase = 0.015

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.01)

    def forward(self, x):
        x1 = self.act(self.norm1(self.fc1(x)))
        s = self.skip(x1)
        x2 = self.act(self.norm2(self.fc2(x1))) + 0.2 * s
        if self.training:
            x2 = self.dropout(x2)

        v = self.val(x2)
        a = self.adv(x2)
        y = self.reg(x2).squeeze(1)  # retorno/Δpreço previsto

        q = v + (a - a.mean(dim=1, keepdim=True))
        return q, y

    # 🌿 poda simbiótica
    def aplicar_poda(self, limiar_base=0.002):
        total = sum(p.numel() for p in self.parameters())
        podadas = 0
        with torch.no_grad():
            for n, p in self.named_parameters():
                if "weight" in n:
                    limiar = limiar_base * (1 + p.abs().mean().item() * 5)
                    mask = p.abs() > limiar
                    podadas += torch.numel(p) - mask.sum().item()
                    p.mul_(mask)
        return podadas / total

    # 🧬 regeneração simbiótica
    def regenerar_sinapses(self, taxa_poda):
        if taxa_poda > 0.15:
            with torch.no_grad():
                for n, p in self.named_parameters():
                    if "weight" in n:
                        var = torch.var(p)
                        mask = torch.rand_like(p) < 0.03
                        novos = torch.randn_like(p) * (var.sqrt() * 0.5)
                        p.add_(mask.float() * novos)

    # ⚖️ homeostase simbiótica
    def verificar_homeostase(self, media):
        if media is None:
            return
        self.historico.append(media)
        if len(self.historico) < 20:
            return
        m = np.mean(self.historico[-10:])
        if self.media_antiga is not None:
            delta = abs(m - self.media_antiga)
            self.estavel = self.estavel + 1 if delta < self.limiar_homeostase else 0
            if self.estavel >= 15:
                self._reset_norms()
                self.estavel = 0
        self.media_antiga = m

    def _reset_norms(self):
        with torch.no_grad():
            for n, p in self.named_parameters():
                if "weight" in n:
                    p.add_(torch.randn_like(p) * 0.02)
            for m in self.modules():
                if isinstance(m, nn.LayerNorm):
                    m.reset_parameters()


# =========================================================
# 🔧 Criação do modelo e otimizador
# =========================================================
def criar_modelo(device):
    modelo = RedeAvancada().to(device)
    alvo = RedeAvancada().to(device)
    alvo.load_state_dict(modelo.state_dict())
    opt = optim.AdamW(modelo.parameters(), lr=LR, weight_decay=1e-4, amsgrad=True)
    return modelo, alvo, opt
