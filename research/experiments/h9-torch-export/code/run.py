"""H9: torch export of ts_fb_C1 vs the sklearn reference, on R3 (see ../protocol.md)."""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP.parents[1] / "src"))
import fp  # noqa: E402
import riemann  # noqa: E402
import torch_fp  # noqa: E402

torch.set_num_threads(1)
d = fp.load_eval()
X, t, run = d["X"], d["person"], d["run"]
tr, te = run <= 1, run == 2
out = {}
for dt in (np.float64, np.float32):
    model, ref = torch_fp.fit_export(X[tr], t[tr], riemann.BANDS["fb"], C=1.0, dtype=dt)
    P_ref = ref(X[te])
    t0 = time.time()
    with torch.inference_mode():
        P_t = torch.softmax(model(torch.from_numpy(X[te])), 1).double().numpy()
    dt_s = time.time() - t0
    nbytes = sum(b.numel() * b.element_size() for b in model.buffers()) + \
        sum(p.numel() * p.element_size() for p in model.parameters())
    r = {"max_abs_dp": float(np.abs(P_ref - P_t).max()),
         "argmax_agree": float((P_ref.argmax(1) == P_t.argmax(1)).mean()),
         "bal_acc_ref": fp.metrics(P_ref, t[te])["bal_acc"],
         "bal_acc_torch": fp.metrics(P_t, t[te])["bal_acc"],
         "infer_s_840": round(dt_s, 2), "state_mb": round(nbytes / 1e6, 2)}
    out[np.dtype(dt).name] = r
    fp.log(f"{np.dtype(dt).name}: {r}")
(EXP / "results").mkdir(exist_ok=True)
(EXP / "results" / "equivalence.json").write_text(json.dumps(out, indent=2))
