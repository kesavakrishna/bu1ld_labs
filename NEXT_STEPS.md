# Next steps

Ideas for a second month. Nothing here was run for the week-1 result; see [FREEZE.md](FREEZE.md) for what was.

## Moved out of the frozen protocol

- **Second failure definition.** Divide the error by the *starting* wave size instead of the current one. Dropped because, on dev, failure wasn't driven by the wave shrinking (at failure the true wave was still 96–98% of its step-20 size).
- **`eff_rank` over a window of steps.** Measure effective rank on the matrix of the last 10 steps' activation summaries, rather than a single step's activations.
- **Threshold stability.** Resample the calibration set about 200 times, recompute thresholds, and check whether the verdict flips.

## Ideas that came up along the way

- **Compare each rollout against its own start, not the population.** Scores currently measure distance from the normal range of *all* trajectories at that step, which is wide. A rollout can drift a long way from its own truth and still look normal. Measuring change relative to each rollout's own first few steps might be far more sensitive. It needs no ground truth.
- **Other hook layers.** Read layers 1, 2 and 4 as well as 3, to see whether any depth carries an earlier signal.
- **Signals aimed at blow-ups.** On dev, output-side signals spiked only after the 10% line, as rollouts began to blow up. Predicting blow-up, rather than 10% error, may be the more natural target.
- **Harder or longer-lived physics.** Decaying Burgers' dissipates errors, so most rollouts drift to a plateau. A forced Burgers' equation would keep shocks forming for the whole rollout.
- **Other architectures.** The hook works on any PyTorch layer, so the same harness could be pointed at a U-Net or transformer surrogate.
