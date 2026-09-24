"""
Day 1 checks for the simulator (PROJECT_PLAN.md, Section 6).

Simulates a small batch of fresh waves (not the dataset) and checks:

  1. Energy never goes up.                  pass / fail
  2. 256 and 1024 grid points agree.        pass / fail
  3. Shocks form: steepness spikes.         reported, and shown in the figure
  4. How much the wave changes per step.    reported, to guide the model's step size
  5. How much energy is left at the end.    reported (see PROJECT_PLAN.md, Section 12)

Prints a summary and saves a figure to <results_dir>/day1_solver_checks.png. If check 1
or 2 fails, the script stops with an error so run_all.sh goes no further.

    python src/check_solver.py --config configs/full.yaml
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw straight to a file; no window needed
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from solver import energy, largest_energy_rise, random_initial_conditions, simulate


def relative_size(a, b):
    """‖a‖ / ‖b‖ for each wave, where the last axis is the grid."""
    return torch.linalg.vector_norm(a, dim=-1) / torch.linalg.vector_norm(b, dim=-1)


def steepness(u):
    """The steepest slope anywhere on each wave. A sudden spike means a shock has formed."""
    n_points = u.shape[-1]
    slope = (torch.roll(u, shifts=-1, dims=-1) - u) * n_points  # (next point - this point) / gap
    return slope.abs().max(dim=-1).values


def save_figure(u, energy_ratio, steep, resolution_error, max_resolution_error, path):
    """Four panels: one wave over time, energy, steepness, and the resolution check."""
    plt.rcParams.update({
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#c3c2b7",
        "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.6,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelcolor": "#52514e", "xtick.color": "#898781", "ytick.color": "#898781",
    })
    blue = "#2a78d6"
    steps = np.arange(u.shape[1])
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)

    # Panel 1: one trajectory at a few moments, light (early) to dark (late).
    ax = axes[0, 0]
    n_saves = u.shape[1] - 1
    moments = [0, n_saves // 40, n_saves // 10, n_saves // 4, n_saves]
    shades = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
    x = np.arange(u.shape[2]) / u.shape[2]
    for step, shade in zip(moments, shades):
        ax.plot(x, u[0, step], color=shade, linewidth=2, label=f"step {step}")
    ax.set(title="One wave over time: it steepens, then flattens", xlabel="position around the ring", ylabel="u")
    ax.legend(frameon=False)

    # Panel 2: energy, as a fraction of each wave's starting energy.
    ax = axes[0, 1]
    ax.plot(steps, energy_ratio.T, color=blue, linewidth=1, alpha=0.5)
    ax.set(title="Energy: must only go down", xlabel="step", ylabel="energy ÷ starting energy")

    # Panel 3: steepness.
    ax = axes[1, 0]
    ax.plot(steps, steep.T, color=blue, linewidth=1, alpha=0.5)
    ax.set(title="Steepness: a spike means a shock formed", xlabel="step", ylabel="steepest slope")

    # Panel 4: disagreement between the 256-point and 1024-point runs.
    ax = axes[1, 1]
    ax.plot(steps, 100 * resolution_error.T, color=blue, linewidth=1, alpha=0.5)
    ax.axhline(100 * max_resolution_error, color="#52514e", linewidth=1, linestyle="--")
    ax.text(steps[-1], 100 * max_resolution_error, "allowed", color="#52514e", ha="right", va="bottom")
    ax.set(title="256 vs 1024 grid points: must stay under the line", xlabel="step", ylabel="difference (%)")

    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Day 1 checks for the simulator.")
    parser.add_argument("--config", default="configs/full.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    sim = config["simulator"]
    ic = config["initial_conditions"]
    checks = config["solver_checks"]

    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    factor = checks["fine_points"] // sim["n_points"]  # 4 for 1024 vs 256

    # Identical starting waves on both grids: build them on the fine grid, then keep every
    # `factor`-th point. Those points sit at exactly the positions of the normal grid.
    u0_fine = random_initial_conditions(
        checks["n_trajectories"], checks["fine_points"], ic["n_modes"], ic["decay"], ic["amplitude"], checks["seed"]
    )
    u0 = u0_fine[:, ::factor]

    start = time.time()
    u = simulate(u0.to(device), sim["viscosity"], sim["dt_solver"], sim["steps_per_save"], sim["n_saves"])
    normal_seconds = time.time() - start

    # Finer grids need a smaller time step to stay stable, so shrink it by the same factor.
    # Taking `factor` times as many steps per save keeps the snapshots at the same moments.
    start = time.time()
    u_fine = simulate(
        u0_fine.to(device), sim["viscosity"], sim["dt_solver"] / factor, sim["steps_per_save"] * factor, sim["n_saves"]
    )
    fine_seconds = time.time() - start
    u_fine_on_normal_grid = u_fine[:, :, ::factor]

    # Check 1: energy never goes up.
    rise = largest_energy_rise(u)
    energy_ok = rise <= checks["max_energy_rise"]

    # Check 2: the two grids agree at the points they share.
    resolution_error = relative_size(u - u_fine_on_normal_grid, u_fine_on_normal_grid)  # [n_traj, n_steps]
    worst_error = resolution_error.max().item()
    worst_step = resolution_error.max(dim=0).values.argmax().item()
    resolution_ok = worst_error <= checks["max_resolution_error"]

    # Reported numbers.
    change = relative_size(u[:, 1:] - u[:, :-1], u[:, :-1])  # change per step, [n_traj, n_steps - 1]
    e = energy(u)
    energy_ratio = e / e[:, :1]
    steep = steepness(u)
    peak_step = steep.argmax(dim=1).float()
    peak_vs_start = steep.max(dim=1).values / steep[:, 0]

    def mark(ok):
        return "PASS" if ok else "FAIL"

    print(f"\nDay 1 simulator checks ({checks['n_trajectories']} trajectories, {device})")
    print(f"  Simulation time: {sim['n_points']} points {normal_seconds:.1f} s, {checks['fine_points']} points {fine_seconds:.1f} s\n")
    print(f"  [{mark(energy_ok)}] Energy never goes up        largest rise {rise:.1e} (allowed {checks['max_energy_rise']:.0e})")
    print(f"  [{mark(resolution_ok)}] 256 and 1024 grids agree    worst difference {100 * worst_error:.2f}% at step {worst_step} (allowed {100 * checks['max_resolution_error']:.0f}%)\n")
    print(f"  Change per step      first 10 steps: median {100 * change[:, :10].median():.1f}%   all steps: median {100 * change.median():.1f}%   last step: median {100 * change[:, -1].median():.1f}%")
    print(f"  Energy left at end   median {100 * energy_ratio[:, -1].median():.1f}% (range {100 * energy_ratio[:, -1].min():.1f}% to {100 * energy_ratio[:, -1].max():.1f}%)")
    print(f"  Steepness            peaks at step {peak_step.median():.0f} (median), {peak_vs_start.median():.0f}x steeper than at the start")

    results_dir = Path(config["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    figure_path = results_dir / "day1_solver_checks.png"
    save_figure(u, energy_ratio, steep, resolution_error, checks["max_resolution_error"], figure_path)
    print(f"\n  Figure saved to {figure_path}")

    if not (energy_ok and resolution_ok):
        sys.exit("Day 1 checks failed: the simulator can't be trusted. Stopping here.")


if __name__ == "__main__":
    main()
