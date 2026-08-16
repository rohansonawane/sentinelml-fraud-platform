# GPU model comparison

**Not benchmarked on this hardware.**

Clone this repo on a Linux NVIDIA machine, then:

```bash
make gpu-setup && make gpu-check
make gpu-data
make gpu-train
make gpu-eval
```

`make gpu-eval` prints the champion from `artifacts/gpu/metrics.json`. Until that command has been run, there are no large-scale numbers to report.

Do not copy Mac smoke PR-AUC into this page.
