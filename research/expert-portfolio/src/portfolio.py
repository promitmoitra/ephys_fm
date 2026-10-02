"""Loop C: N-stream reliability-weighted log-linear combiner, and its locked evaluation."""

import sys
from pathlib import Path

import numpy as np
import torch

SRC = Path(__file__).resolve().parent
REPO = SRC.parents[2]
sys.path.insert(0, str(REPO / "research" / "expert-weights" / "src"))
from combine import Combiner  # noqa: E402

OUT = REPO / "outputs" / "t2-portfolio"


class PortfolioLogLinear(Combiner):
    """z = Σ_s w_s·log p_s + (c0 + c1·(rel − 0.5))·log p_riemann + [0, b];
    experts = [*neural_streams, riemann_key, rel_key]. With one neural stream this is
    loop B's RelLogLinear (same initialisation and penalty)."""

    def init(self):
        S = len(self.experts) - 2
        th = {"w": torch.full((S,), 0.8, dtype=torch.float64),
              "c": torch.tensor([0.4, 0.0], dtype=torch.float64),
              "b": torch.zeros(1, dtype=torch.float64)}
        return th

    def penalty(self, theta, theta0):
        return sum(((theta[k] - theta0[k]) ** 2).sum() for k in theta)

    def combine(self, F, theta):
        *neural, cls, rel = self.experts
        z = sum(w * F[e] for w, e in zip(theta["w"], neural))
        z = z + (theta["c"][0] + theta["c"][1] * (F[rel] - 0.5)) * F[cls]
        z = z + torch.cat([torch.zeros_like(theta["b"]), theta["b"]])
        return torch.log_softmax(z, -1)
