"""§14 sweep autorizado: KERNEL_LENGTHSCALE ∈ {0.5, 1.0, 2.0} × audited.
Escolha: menor pinball médio (por slice e agregado) no replay offline das
janelas CAL (§16: escolha por validação CAL, depois FREEZE). Reproduz o
pipeline do sqr_jc_calibrate.py variando apenas o lengthscale."""
import sys, os, glob, json
sys.path.insert(0, "src"); sys.path.insert(0, "examples")
import numpy as np
import sqr_jc_calibrate as cal
from sqr_jc.kernel_models import MultiQuantileKernelModel
from sqr_jc.joint_state import JOINT_STATE_DIM

RUNS = sorted(glob.glob("/home/eliothluy/Documentos/Jussi_SBRC/results/sqr_jc_final/03_calibration/cal_runs/cal_*"))
OUT = "/home/eliothluy/Documentos/Jussi_SBRC/results/sqr_jc_final/03_calibration/ls_sweep"

def replay(ls):
    models = tuple(MultiQuantileKernelModel(dim=JOINT_STATE_DIM + 3, length_scale=ls,
                                            memory_size=cal.B_MEMORY) for _ in range(3))
    pin = np.zeros(3); n = 0
    for run in RUNS:
        windows = cal._load_windows(run) if hasattr(cal, "_load_windows") else None
        if windows is None:
            return None
        for w in windows:
            x = w["x"]; ys = w["y"]
            for i in range(3):
                f = models[i].predict(x)
                for tau, head in ((0.25, "q25"), (0.75, "q75")):
                    pass  # pinball nas heads aux via pred
                pin[i] += cal._pinball(ys[i], f, models[i]) if hasattr(cal, "_pinball") else 0
            n += 1
    return pin / max(n, 1), n

if __name__ == "__main__":
    # introspecção: precisamos das funções reais do calibrador
    names = [n for n in dir(cal) if not n.startswith("__")]
    print("cal exports:", [n for n in names if "load" in n or "window" in n or "pin" in n or "replay" in n])
