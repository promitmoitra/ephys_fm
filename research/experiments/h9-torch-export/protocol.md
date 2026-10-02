# H9: a shippable torch fingerprint that reproduces `ts_fb_C1`

**Status:** engineering check (pass/fail, not a hypothesis about data).

`submission.py` must run with torch + braindecode only; the handoff asks for weights as tensors,
no sklearn / pyriemann objects at inference. `research/src/torch_fp.py` exports the fitted
pipeline: per-band zero-phase filter as a (T × T) matrix (filtfilt on a fixed window is
linear), OAS in torch, tangent space via `eigh`, StandardScaler + LR folded into one Linear.

**Test (dev data only):** fit on R1–R2, predict R3 with both paths.
Pass if: max |Δp| < 1e-4 (float64 buffers) and identical argmax on all 840 windows; report the
float32 variant too, and CPU time and buffer size for 2,520 windows.
