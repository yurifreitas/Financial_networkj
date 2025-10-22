# ==========================================
# 🧠 EtherSym Finance — Rede Dueling DQN + Regressão de Preço (Avançada)
# ==========================================
# - Cabeças duplas: Ações discretas (Q-values) + Retorno contínuo (regressão)
# - Poda, regeneração e homeostase simbiótica
# - Compatível com main_v8f_turbo_simbiotico.py
# - Estrutura viva: autoestabiliza e se adapta ao longo do treino
# ==========================================

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


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
        self.val = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
        self.adv = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, n_actions))
        self.reg = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))

        # Buffers simbióticos internos
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
        y_lin = self.reg(x2).squeeze(1)
        y = torch.tanh(y_lin)  # Bound suave

        q = v + (a - a.mean(dim=1, keepdim=True))
        return q, y


def criar_modelo(device, lr=1e-4):
    modelo = RedeAvancada().to(device)
    alvo = RedeAvancada().to(device)
    alvo.load_state_dict(modelo.state_dict())
    opt = optim.AdamW(modelo.parameters(), lr=lr, weight_decay=1e-4, amsgrad=True)
    return modelo, alvo, opt

