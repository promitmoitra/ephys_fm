# P0: REVE offline feasibility: **GO**

Raw numbers: `results.json` (4 CPU threads, 2026-10-02).

| Check | Result |
|---|---|
| Offline load (`HF_HUB_OFFLINE=1`, positions from `REVE_POSITIONS_PATH`) | ok |
| Constructor | plain `REVE.from_pretrained("brain-bzh/reve-base")` fails (`ValueError: n_outputs not specified`); `from_pretrained(..., n_outputs=2, n_chans=27, n_times=800, sfreq=200)` works. Saved as `reve_kwargs.json` next to the positions file, which `load_reve_encoder` reads |
| Pretrained weights actually loaded | 139 / 140 checkpoint tensors identical in the model; the extra `final_layer` (randomly initialised) is unused because the probe reads `return_features`; `cls_query_token` (attention pooling) is not loaded and not used |
| Parameters | 69.4 M |
| Channels missing from the position bank | Dreyer: none (27); BNCI: none (22) |
| Features | `(B, 27, 4, 512)` for a 4-s window at 200 Hz (4 patches of 200 samples, overlap 20) |
| Embedding time | 96.7 s / 1,000 windows → ~34 min for all 20,792 Dreyer windows |
| Download (first time) | 145 s (277 MB weights + 60 KB position bank) |

**Decision: GO** for S1 (offline load ok, no missing channels, full-Dreyer embedding ≈ 34 min ≤ 120).

Notes for later tasks:
- The geometry kwargs only shape REVE's unused output head; features work for any channel count
  (the encoder embeds per channel with explicit positions). They are still the shipped
  constructor call.
- Pooling: mean over channels × patches, as the spec says. REVE's checkpoint also carries an
  attention-pooling query (`cls_query_token`), which would be an alternative if mean pooling
  turns out weak.
