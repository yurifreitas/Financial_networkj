# =========================================================
# 🌌 EtherSym Finance — run_infinite_parallel.py
# =========================================================
# - Executa várias instâncias simbióticas em paralelo
# - Cada instância tem log próprio e pode usar GPU diferente
# - Ideal para execuções 24/7 em cluster local
# =========================================================

import os, sys, time, subprocess, torch

# =========================================================
# ⚙️ Configuração geral
# =========================================================
INSTANCIAS = 4           # número de instâncias paralelas
BATCH = 3                # episódios por lote
SLEEP = 2                # pausa entre lotes (segundos)
INTERVALO_START = 5      # intervalo entre inicializações
SCRIPT = "main_v10_infinite_persistente.py"

# =========================================================
# 🔍 Detecta GPUs disponíveis
# =========================================================
gpu_count = torch.cuda.device_count()
print(f"🎛️ Detectadas {gpu_count} GPU(s) disponíveis")

# =========================================================
# 🧠 Inicia instâncias simbióticas paralelas
# =========================================================
processos = []

for i in range(INSTANCIAS):
    gpu_id = i % max(1, gpu_count)
    log_name = f"logs/treino_simbiotico_{i+1}.log"
    os.makedirs("logs", exist_ok=True)

    cmd = [
        sys.executable,
        SCRIPT,
        "--batch", str(BATCH),
        "--sleep", str(SLEEP)
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"🚀 Iniciando instância simbiótica {i+1}/{INSTANCIAS} na GPU {gpu_id}")
    print(f"📄 Log: {log_name}")

    with open(log_name, "w") as log_file:
        p = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env)
        processos.append(p)

    time.sleep(INTERVALO_START)

# =========================================================
# 🧩 Monitor simbiótico das execuções
# =========================================================
try:
    while True:
        ativos = sum(p.poll() is None for p in processos)
        print(f"🧬 Instâncias ativas: {ativos}/{INSTANCIAS}")
        if ativos < INSTANCIAS:
            print("⚠️ Alguma instância terminou — reiniciando...")
            for i, p in enumerate(processos):
                if p.poll() is not None:
                    gpu_id = i % max(1, gpu_count)
                    log_name = f"logs/restart_{i+1}.log"
                    with open(log_name, "w") as log_file:
                        new_p = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env)
                        processos[i] = new_p
                        print(f"♻️ Reiniciada instância {i+1} na GPU {gpu_id}")
        time.sleep(60)
except KeyboardInterrupt:
    print("\n🧹 Encerrando todas as instâncias simbióticas...")
    for p in processos:
        p.terminate()
    print("✅ Todas as execuções foram finalizadas com segurança.")
