"""
Simulator for the 1D Burgers' equation, and dataset generation.

    du/dt + u * du/dx = viscosity * d²u/dx²

The wave u lives on a ring: x runs from 0 to 1, and x = 1 is the same point as x = 0.
PROJECT_PLAN.md, Section 2, explains what the equation means.

How the simulator works
-----------------------
- The wave is stored as Fourier coefficients: how much of each sine wave adds up to it.
  In that form, taking a derivative d/dx is exact: multiply coefficient number m by i·2π·m.
- Time moves forward with RK4, a standard, accurate way to take small steps.
- The u * du/dx term creates wiggles finer than the grid can hold, and they come back as
  fake ripples. The "2/3 rule" stops this by zeroing the finest third of the coefficients.
- All trajectories are simulated together in one batch, which is what makes it fast.
  A batch of waves has shape [n_trajectories, n_points].

Generate the dataset with:

    python src/solver.py --config configs/full.yaml
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import yaml


def random_initial_conditions(n_trajectories, n_points, n_modes, decay, amplitude, seed):
    """
    Make random, smooth starting waves.

    Each wave is a sum of the first `n_modes` sine waves, each with a random size and a
    random sideways shift. Finer sine waves are made smaller (size divided by mode**decay)
    so the wave starts smooth. Every wave is then rescaled so its highest point is `amplitude`.

    Returns a float64 tensor of shape [n_trajectories, n_points].
    """
    generator = torch.Generator().manual_seed(seed)
    sizes = torch.randn(n_trajectories, n_modes, generator=generator, dtype=torch.float64)
    shifts = 2 * torch.pi * torch.rand(n_trajectories, n_modes, generator=generator, dtype=torch.float64)

    modes = torch.arange(1, n_modes + 1, dtype=torch.float64)  # 1, 2, ..., n_modes
    sizes = sizes / modes**decay

    x = torch.arange(n_points, dtype=torch.float64) / n_points  # grid points in [0, 1)

    # Build every sine wave for every trajectory at once, then add them up.
    # The [:, :, None]-style indexing lines up the shapes as [trajectory, mode, point].
    sine_waves = sizes[:, :, None] * torch.sin(
        2 * torch.pi * modes[None, :, None] * x[None, None, :] + shifts[:, :, None]
    )
    u0 = sine_waves.sum(dim=1)

    peak = u0.abs().max(dim=1, keepdim=True).values
    return amplitude * u0 / peak


def simulate(u0, viscosity, dt_solver, steps_per_save, n_saves):
    """
    Run the simulator forward from the starting waves `u0`.

    Args:
        u0:             starting waves, shape [n_trajectories, n_points], on CPU or GPU
        viscosity:      smoothing strength
        dt_solver:      the tiny time step the simulator takes
        steps_per_save: tiny steps between saved snapshots
        n_saves:        snapshots to save after the starting wave

    Returns:
        float32 CPU tensor of shape [n_trajectories, n_saves + 1, n_points].
        Snapshot 0 is the starting wave.
    """
    n_points = u0.shape[1]

    # rfft stores one coefficient per sine wave that fits 0, 1, ..., n_points/2 times around the ring.
    m = torch.arange(n_points // 2 + 1, device=u0.device, dtype=torch.float64)
    k = 2 * torch.pi * m    # d/dx multiplies coefficient m by i·k
    keep = (m < n_points / 3).to(torch.float64)  # the 2/3 rule: 1 = keep, 0 = zero out

    def rate_of_change(u_hat):
        """du/dt, in Fourier coefficients."""
        u = torch.fft.irfft(u_hat, n=n_points)  # coefficients -> values on the grid
        # u * du/dx is the same as d/dx (u²/2), which is simpler to compute here.
        half_u_squared_hat = torch.fft.rfft(0.5 * u * u) * keep
        steepening = -1j * k * half_u_squared_hat  # -d/dx (u²/2)
        smoothing = -viscosity * k**2 * u_hat      # viscosity · d²u/dx²
        return steepening + smoothing

    u_hat = torch.fft.rfft(u0.to(torch.float64)) * keep
    snapshots = [torch.fft.irfft(u_hat, n=n_points).float().cpu()]

    for _ in range(n_saves):
        for _ in range(steps_per_save):
            # One RK4 step: estimate the rate of change four times across the step,
            # then move forward by a weighted average of the four estimates.
            rate1 = rate_of_change(u_hat)
            rate2 = rate_of_change(u_hat + 0.5 * dt_solver * rate1)
            rate3 = rate_of_change(u_hat + 0.5 * dt_solver * rate2)
            rate4 = rate_of_change(u_hat + dt_solver * rate3)
            u_hat = u_hat + (dt_solver / 6) * (rate1 + 2 * rate2 + 2 * rate3 + rate4)
        snapshots.append(torch.fft.irfft(u_hat, n=n_points).float().cpu())

    return torch.stack(snapshots, dim=1)


def energy(u):
    """Average of u² around the ring, for every wave in `u` (the last axis must be the grid)."""
    return (u.double() ** 2).mean(dim=-1)


def largest_energy_rise(u):
    """
    The biggest step-to-step energy increase, as a fraction of the energy before the step.
    Energy can only fall for this equation, so this should be about 0.

    `u` has shape [n_trajectories, n_steps, n_points].
    """
    e = energy(u)
    rise = (e[:, 1:] - e[:, :-1]) / e[:, :-1]
    return rise.max().item()


def main():
    parser = argparse.ArgumentParser(description="Generate the Burgers' equation dataset.")
    parser.add_argument("--config", default="configs/full.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    sim = config["simulator"]
    ic = config["initial_conditions"]
    data = config["data"]

    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

    # Train, calibration and dev come from one draw using `seed`. Test comes from a separate
    # draw using `test_seed`, so it could be kept unseen until the frozen run (FREEZE.md).
    # A trajectory's position in the combined batch is its ID; later code uses IDs to prove
    # the splits never overlap.
    split_sizes = {"train": data["n_train"], "calibration": data["n_calibration"], "dev": data["n_dev"]}
    wave_settings = (sim["n_points"], ic["n_modes"], ic["decay"], ic["amplitude"])
    u0 = torch.cat([
        random_initial_conditions(sum(split_sizes.values()), *wave_settings, data["seed"]),
        random_initial_conditions(data["n_test"], *wave_settings, data["test_seed"]),
    ])
    split_sizes["test"] = data["n_test"]
    n_total = len(u0)

    print(f"Simulating {n_total} trajectories on {device} ...")
    start = time.time()
    u = simulate(u0.to(device), sim["viscosity"], sim["dt_solver"], sim["steps_per_save"], sim["n_saves"])
    print(f"  finished in {time.time() - start:.1f} s")

    # Refuse to save broken data.
    rise = largest_energy_rise(u)
    if rise > config["solver_checks"]["max_energy_rise"]:
        raise RuntimeError(
            f"Energy went up by {rise:.2e} somewhere, so the simulator is unstable. Try a smaller dt_solver."
        )

    data_dir = Path(config["data_dir"]).expanduser()
    data_dir.mkdir(parents=True, exist_ok=True)

    first = 0
    for split, size in split_sizes.items():
        path = data_dir / f"{split}.npz"
        np.savez(
            path,
            u=u[first : first + size].numpy(),      # [n_trajectories, n_saves + 1, n_points]
            ids=np.arange(first, first + size),
            seed=data["test_seed"] if split == "test" else data["seed"],
            dt=sim["dt_solver"] * sim["steps_per_save"],  # time between saved snapshots
            config=yaml.safe_dump(config),          # every setting used, for traceability
        )
        print(f"  saved {size:>4} trajectories to {path}")
        first += size


if __name__ == "__main__":
    main()
