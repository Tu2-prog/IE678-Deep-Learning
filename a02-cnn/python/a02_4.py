# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
#Anh Tu Duong Nguyen (anguyea, 115931)
#
#Anh-Nhat Nguyen (anhnnguy, 2034311)

# %%
import pandas as pd
import torch
from itertools import product
from IPython.display import display

from a02_functions import SimpleCNN, train_model
from a02_helper import get_raw_data, count_model_params

# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# # Task 4: Exploration

# %%
# TODO: your code here
import matplotlib.pyplot as plt
from torch import nn

data = get_raw_data()

# %% [markdown]
# ## Helpers

# %%
def compute_linear_in(channels=25, kernel_size=3, stride=2, padding=1, input_len=40):
    """Use a dummy forward pass through the conv layers to find the flattened size.
    More robust than manual math — adapts automatically if the architecture changes."""
    conv_layers = nn.Sequential(
        nn.Conv1d(1, channels, kernel_size, stride=stride, padding=padding),
        nn.ReLU(),
        nn.Conv1d(channels, channels, kernel_size, stride=stride, padding=padding),
        nn.ReLU(),
        nn.MaxPool1d(2, 2),
        nn.Conv1d(channels, channels, kernel_size, stride=stride, padding=padding),
        nn.ReLU(),
        nn.Conv1d(channels, channels, kernel_size, stride=stride, padding=padding),
        nn.ReLU(),
        nn.MaxPool1d(2, 2),
    )
    with torch.no_grad():
        dummy = torch.zeros(1, 1, input_len)
        out = conv_layers(dummy)
    return out.flatten(start_dim=1).shape[1]


def run_experiment(label, subtask, model_kw, train_kw, data):
    """Train SimpleCNN with given hyperparameters and return metrics."""
    defaults = dict(channels=25, kernel_size=3, stride=2, padding=1)
    hp = {**defaults, **model_kw}
    hp["linear_in"] = compute_linear_in(
        channels=hp["channels"], kernel_size=hp["kernel_size"],
        stride=hp["stride"], padding=hp["padding"],
    )
    # Seed before both model init AND DataLoader creation for fully reproducible runs
    torch.manual_seed(0)
    model = SimpleCNN(**hp)
    torch.manual_seed(0)  # re-seed so DataLoader shuffle is identical across experiments
    results = train_model(data, model, **train_kw)
    return {
        "subtask": subtask,
        "config":    label,
        "n_params":  count_model_params(model),
        "test_acc":  results["test_acc"][-1],
        "val_acc":   results["val_acc"][-1],
        "train_acc": results["train_acc"][-1],
    }

# %% [markdown]
# ## Experiment Configurations
#
# Each tuple: (label, subtask, model_kwargs, train_kwargs).
# `✓` marks the default baseline value for easy reference.

# %%
configs = [
    # ── 4a: learning rate ────────────────────────────────────────────────────
    ("lr=1e-3",       "4a", {},                              {"lr": 1e-3, "epochs": 200}),
    ("lr=1e-2 ✓",     "4a", {},                              {"lr": 1e-2}),
    ("lr=1e-1",       "4a", {},                              {"lr": 1e-1}),
    # ── 4a: batch size ───────────────────────────────────────────────────────
    ("bs=16",         "4a", {},                              {"batch_size": 16}),
    ("bs=64 ✓",       "4a", {},                              {"batch_size": 64}),
    ("bs=256",        "4a", {},                              {"batch_size": 256}),
    # ── 4b: kernel size (padding adjusted to keep spatial sizes reasonable) ──
    ("k=1,p=0",       "4b", {"kernel_size": 1, "padding": 0}, {}),
    ("k=3,p=1 ✓",     "4b", {},                              {}),
    ("k=5,p=2",       "4b", {"kernel_size": 5, "padding": 2}, {}),
    # ── 4c: number of channels ───────────────────────────────────────────────
    ("ch=5",          "4c", {"channels": 5},                 {}),
    ("ch=25 ✓",       "4c", {},                              {}),
    ("ch=50",         "4c", {"channels": 50},                {}),
    # ── 4d: stride ───────────────────────────────────────────────────────────
    ("stride=1",      "4d", {"stride": 1},                   {}),
    ("stride=2 ✓",    "4d", {},                              {}),
]

# %% [markdown]
# ## Run All Experiments

# %%
rows = []
for label, subtask, model_kw, train_kw in configs:
    print(f"\n{'─'*50}\n[{subtask}] {label}")
    row = run_experiment(label, subtask, model_kw, train_kw, data)
    rows.append(row)
    print(f"  test_acc={row['test_acc']:.1f}%  val_acc={row['val_acc']:.1f}%  n_params={row['n_params']}")

# %% [markdown]
# ## Summary Table

# %%
df = pd.DataFrame(rows)
display(df)

# %% [markdown]
# ## Visualizations

# %%
subtask_titles = {
    "4a": "4a — Learning Rate & Batch Size",
    "4b": "4b — Kernel Size",
    "4c": "4c — Number of Channels",
    "4d": "4d — Stride",
}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for ax, (st, title) in zip(axes, subtask_titles.items()):
    sub = df[df["subtask"] == st].reset_index(drop=True)
    x = range(len(sub))
    width = 0.25
    ax.bar([i - width for i in x], sub["train_acc"], width=width, label="train_acc", alpha=0.8, color="steelblue")
    ax.bar([i         for i in x], sub["val_acc"],   width=width, label="val_acc",   alpha=0.8, color="orange")
    ax.bar([i + width for i in x], sub["test_acc"],  width=width, label="test_acc",  alpha=0.8, color="green")
    ax.set_xticks(list(x))
    ax.set_xticklabels(sub["config"], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.axhline(y=90, color="red", linestyle="--", linewidth=0.8, label="90% target")

plt.suptitle("Task 4: Hyperparameter Exploration", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()

# %%
# Parameter count comparison (relevant for 4b, 4c, 4d)
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

for ax, st, title in zip(axes, ["4b", "4c", "4d"],
                          ["4b — Kernel Size", "4c — Channels", "4d — Stride"]):
    sub = df[df["subtask"] == st].reset_index(drop=True)
    ax.bar(range(len(sub)), sub["n_params"], color="mediumpurple", alpha=0.8)
    ax.set_xticks(range(len(sub)))
    ax.set_xticklabels(sub["config"], rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("# Parameters")
    ax.set_title(title)

plt.suptitle("Parameter Count by Architectural Hyperparameter", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.show()