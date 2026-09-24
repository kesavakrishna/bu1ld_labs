"""
Train the FNO to predict one step ahead (Day 2).

Trains one model per seed, saves each one, and reports the single-step error on the
dev split. Day 2 passes when that error is under 1%; otherwise the script stops with an
error, so run_all.sh goes no further. The test split is never touched here.

Two deliberate choices:
  * We train on single steps only, never on rollouts. Training on rollouts would make
    the model steadier, and we want it to drift, because drift is what we study.
  * Each seed changes the starting weights and the order the data is shown. The dataset
    stays the same, so the three runs differ only in the training randomness.

    python src/train.py --config configs/full.yaml
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from model import FNO1d


def load_pairs(path, pairs_per_trajectory, seed):
    """
    Load a dataset file and turn it into (wave now, wave one step later) pairs.

    `pairs_per_trajectory` takes that many random steps from each trajectory; pass None
    to use every step. Returns two tensors shaped [n_trajectories, pairs, n_points], so
    callers can either flatten them or keep track of which step each pair came from.
    """
    u = torch.from_numpy(np.load(path)["u"])  # [n_trajectories, n_steps, n_points]
    n_trajectories, n_steps, _ = u.shape

    if pairs_per_trajectory is None:
        starts = torch.arange(n_steps - 1).expand(n_trajectories, n_steps - 1)
    else:
        generator = torch.Generator().manual_seed(seed)
        starts = torch.randint(n_steps - 1, (n_trajectories, pairs_per_trajectory), generator=generator)

    trajectories = torch.arange(n_trajectories)[:, None]
    return u[trajectories, starts], u[trajectories, starts + 1]


def relative_error(prediction, target):
    """Size of the mistake divided by size of the true wave, for each wave separately."""
    return torch.linalg.vector_norm(prediction - target, dim=-1) / torch.linalg.vector_norm(target, dim=-1)


@torch.no_grad()
def errors_on(model, inputs, targets, batch_size):
    """Single-step error for every pair, without training. Returns one number per pair."""
    model.eval()
    errors = [relative_error(model(inputs[i : i + batch_size]), targets[i : i + batch_size])
              for i in range(0, len(inputs), batch_size)]
    model.train()
    return torch.cat(errors)


def train_one_model(config, inputs, targets, seed, device):
    """Train a single model. Returns the model and its average training error per epoch."""
    settings = config["training"]
    torch.manual_seed(seed)

    model = FNO1d(**config["model"]).to(device)
    model.set_normalization(inputs.mean(), inputs.std())

    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    # Cosine decay: the learning rate eases down to almost nothing by the final epoch.
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings["epochs"])

    history = []
    for epoch in range(settings["epochs"]):
        order = torch.randperm(len(inputs), device=device)
        total = torch.zeros((), device=device)

        for start in range(0, len(order), settings["batch_size"]):
            batch = order[start : start + settings["batch_size"]]
            loss = relative_error(model(inputs[batch]), targets[batch]).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.detach() * len(batch)

        schedule.step()
        history.append((total / len(order)).item())
        if (epoch + 1) % 10 == 0:
            print(f"    epoch {epoch + 1:>3}   training error {100 * history[-1]:.3f}%")

    return model, history


def save_figure(history, dev_errors_by_step, path):
    """Left: training progress. Right: where in a trajectory the model struggles."""
    plt.rcParams.update({
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#c3c2b7",
        "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.6,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelcolor": "#52514e", "xtick.color": "#898781", "ytick.color": "#898781",
    })
    shades = ["#2a78d6", "#eb6834", "#1baf7a"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)

    ax = axes[0]
    for (seed, values), shade in zip(history.items(), shades):
        ax.plot(np.arange(1, len(values) + 1), 100 * np.array(values), color=shade, linewidth=2, label=f"seed {seed}")
    ax.set(title="Training progress", xlabel="epoch", ylabel="training error (%)", yscale="log")
    ax.legend(frameon=False)

    ax = axes[1]
    for (seed, errors), shade in zip(dev_errors_by_step.items(), shades):
        median = 100 * np.median(errors, axis=0)
        ax.plot(np.arange(len(median)), median, color=shade, linewidth=2, label=f"seed {seed}")
    ax.set(title="Dev error by step in the trajectory", xlabel="step", ylabel="single-step error (%)", yscale="log")
    ax.legend(frameon=False)

    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train the FNO on single-step prediction.")
    parser.add_argument("--config", default="configs/full.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    settings = config["training"]

    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    data_dir = Path(config["data_dir"])

    # Make GPU training repeatable: the same seed should give the same model every time.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_inputs, train_targets = load_pairs(
        data_dir / "train.npz", settings["pairs_per_trajectory"], config["data"]["seed"]
    )
    train_inputs = train_inputs.flatten(0, 1).to(device)
    train_targets = train_targets.flatten(0, 1).to(device)

    # Every single-step pair in the dev split, kept as [n_trajectories, steps, n_points]
    # so we can also see how the error depends on where we are in a trajectory.
    dev_inputs, dev_targets = load_pairs(data_dir / "dev.npz", None, seed=0)
    n_dev_trajectories, n_dev_steps, _ = dev_inputs.shape
    dev_inputs = dev_inputs.flatten(0, 1).to(device)
    dev_targets = dev_targets.flatten(0, 1).to(device)

    print(f"Device: {device}")
    print(f"{len(train_inputs)} training pairs, {len(dev_inputs)} dev pairs")

    model_dir = Path(config["results_dir"]) / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    history, errors_by_step, dev_error = {}, {}, {}
    for seed in settings["seeds"]:
        print(f"\n  seed {seed}")
        start = time.time()
        model, values = train_one_model(config, train_inputs, train_targets, seed, device)
        print(f"    trained in {time.time() - start:.0f} s ({model.n_parameters():,} learned numbers)")

        errors = errors_on(model, dev_inputs, dev_targets, settings["batch_size"])
        history[seed] = values
        errors_by_step[seed] = errors.reshape(n_dev_trajectories, n_dev_steps).cpu().numpy()
        dev_error[seed] = errors.mean().item()

        path = model_dir / f"model_seed{seed}.pt"
        torch.save({"state_dict": model.state_dict(), "config": config, "seed": seed}, path)
        print(f"    saved to {path}")

    limit = settings["max_single_step_error"]
    print(f"\nDay 2 check: single-step dev error must be under {100 * limit:.0f}%")
    for seed, error in dev_error.items():
        print(f"  [{'PASS' if error < limit else 'FAIL'}] seed {seed}   dev error {100 * error:.3f}%")

    figure_path = Path(config["results_dir"]) / "day2_training.png"
    save_figure(history, errors_by_step, figure_path)
    print(f"\n  Figure saved to {figure_path}")

    if any(error >= limit for error in dev_error.values()):
        sys.exit("Day 2 check failed: a model is not accurate enough. Stopping here.")


if __name__ == "__main__":
    main()
