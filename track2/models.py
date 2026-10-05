"""Neural expert architectures for Track 2 (shared by training and submission tooling)."""

SHALLOW_REF_SFREQ = 250.0          # braindecode ShallowFBCSPNet defaults assume 250 Hz
_SHALLOW_REF = dict(filter_time_length=25, pool_time_length=75, pool_time_stride=15)


def shallow_kwargs(sfreq):
    """ShallowFBCSPNet time constants rescaled from 250 Hz to `sfreq` (same durations)."""
    s = sfreq / SHALLOW_REF_SFREQ
    return {k: max(1, int(round(v * s))) for k, v in _SHALLOW_REF.items()}


def make_model(arch, n_chans, n_outputs, n_times, sfreq):
    if arch == "eegnet":
        from braindecode.models import EEGNet
        return EEGNet(n_chans=n_chans, n_outputs=n_outputs, n_times=n_times)
    if arch == "shallow":
        from braindecode.models import ShallowFBCSPNet
        return ShallowFBCSPNet(n_chans=n_chans, n_outputs=n_outputs, n_times=n_times,
                               final_conv_length="auto", **shallow_kwargs(sfreq))
    raise ValueError(f"unknown arch {arch!r}")
