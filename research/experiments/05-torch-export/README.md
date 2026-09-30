# Experiment 05: torch export of C3 (engineering check)

`research/src/torch_experts.py`: `TangentExperts` (21 per-person ts_C0.1 experts as buffers:
one shared 480 × 420 band-pass-and-crop matrix, per-person Cref^-1/2 and LR weights) and
`C3Combiner` (4 scalars + per-person reliability). Torch only, no sklearn / pyriemann /
scipy at inference.

Result (`results/export.json`, R4–R6, packaged seed-0 EEGNet experts): max |Δp| 2.3e-8 vs
the research pipeline, identical argmax on all 52,920 person × window outputs, oracle 0.921
in both. Run the ts path in **float64**: float32 flips one near-tie in 52,920 (|Δp| 3.4e-5).
Buffers 1.8 MB (float64). Fitting all 21 experts plus the reliability scores takes ~15 s on
one thread.

For `submission.py`: add the ts experts and C3 next to the EEGNet experts;
`p(class | x) = Σ_k p(k | x) · softmax(C3(log p_eeg_k(x), log p_ts_k(x)))`.
