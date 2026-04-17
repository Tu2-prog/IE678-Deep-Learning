# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     custom_cell_magics: kql
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: deep-learning
#     language: python
#     name: python3
# ---

# %% [markdown]
# Anh Tu Duong Nguyen (anguyea, 115931)
#
# Anh-Nhat Nguyen (anhnnguy, 2034311)

# %%
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from itertools import product

from a02_functions import SimpleCNN, train_model
from a02_helper import get_raw_data, count_model_params

# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# # Task 4: Exploration

# %%
data = get_raw_data()

def run_sweep(data, model_fn, configs: list[dict], sweep_name: str = "") -> dict[str, dict]:
    all_results = {}
    for cfg in configs:
        label = ", ".join(f"{k}={v}" for k, v in cfg.items())
        print(f"\n--- [{sweep_name}] Config: {label} ---")
        model = model_fn()
        results = train_model(data, model, **cfg)
        results["num_params"] = count_model_params(model)
        all_results[label] = results
    return all_results


def plot_sweep(all_results: dict, metric: str = "val_acc", eval_every: int = 10, title_prefix: str = ""):
    EPOCH_METRICS = {"train_acc", "val_acc", "train_losses", "val_losses"}
    if metric not in EPOCH_METRICS:
        raise ValueError(f"'{metric}' is a scalar. Use one of: {sorted(EPOCH_METRICS)}")

    loss_key = "val_losses" if "val" in metric else "train_losses"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for label, res in all_results.items():
        epochs = [i * eval_every for i in range(len(res[metric]))]
        axes[0].plot(epochs, res[loss_key], label=label)
        axes[1].plot(epochs, res[metric], label=label)

    axes[0].set(title=f"{title_prefix} Loss", xlabel="Epoch", ylabel="Loss")
    axes[1].set(title=f"{title_prefix} Accuracy", xlabel="Epoch", ylabel="Accuracy (%)")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_final_comparison(all_results: dict, metric: str = "test_acc", title_prefix: str = ""):
    labels = list(all_results.keys())
    values = [res[metric][0] for res in all_results.values()]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, values)
    ax.set(title=f"{title_prefix} — Final {metric} by config", ylabel=metric)
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.show()


def build_summary_table(sweep_results: dict[str, dict]) -> pd.DataFrame:
    """
    sweep_results: { sweep_name: { config_label: result_dict } }
    Returns a DataFrame with one row per config, columns:
      sweep | config | test_acc | val_acc (final) | train_acc (final) | val_loss (final) | train_loss (final)
    """
    rows = []
    for sweep_name, all_results in sweep_results.items():
        for label, res in all_results.items():
            rows.append({
                "sweep":            sweep_name,
                "config":           label,
                "test_acc":         res["test_acc"][0],
                "val_acc_final":    res["val_acc"][-1],
                "train_acc_final":  res["train_acc"][-1],
                "val_loss_final":   res["val_losses"][-1],
                "train_loss_final": res["train_losses"][-1],
                "num_params":      res["num_params"],
            })
    df = pd.DataFrame(rows)
    df = df.sort_values(["sweep", "test_acc"], ascending=[True, False]).reset_index(drop=True)
    return df


# %% [markdown]
# ## Run all sweeps

# %%
sweep_results = {}

# --- Learning rate sweep ---
lrs = np.logspace(-5, 0, 10)
configs = [{"lr": float(lr)} for lr in lrs]
sweep_results["lr"] = run_sweep(data, lambda: SimpleCNN(), configs, sweep_name="lr")
plot_sweep(sweep_results["lr"], metric="val_acc", title_prefix="LR Sweep")
plot_final_comparison(sweep_results["lr"], metric="test_acc", title_prefix="LR Sweep")

# --- Batch size sweep ---
batch_sizes = [2**i for i in range(1, 8)]
configs = [{"batch_size": int(bs)} for bs in batch_sizes]
sweep_results["batch_size"] = run_sweep(data, lambda: SimpleCNN(), configs, sweep_name="batch_size")
plot_sweep(sweep_results["batch_size"], metric="val_acc", title_prefix="Batch Size Sweep")
plot_final_comparison(sweep_results["batch_size"], metric="test_acc", title_prefix="Batch Size Sweep")

# --- Epochs sweep ---
epoch_list = [10, 20, 50, 100, 200, 500]
configs = [{"epochs": int(e)} for e in epoch_list]
sweep_results["epochs"] = run_sweep(data, lambda: SimpleCNN(), configs, sweep_name="epochs")
plot_sweep(sweep_results["epochs"], metric="val_acc", title_prefix="Epochs Sweep")
plot_final_comparison(sweep_results["epochs"], metric="test_acc", title_prefix="Epochs Sweep")


# %%
def run_sweep_class_hyperparameter(data, model_fn, configs: list[dict], sweep_name: str = "") -> dict[str, dict]:
    all_results = {}
    for cfg in configs:
        label = ", ".join(f"{k}={v}" for k, v in cfg.items())
        print(f"\n--- [{sweep_name}] Config: {label} ---")
        model = model_fn(**cfg)          # pass config to constructor
        results = train_model(data, model)  # no **cfg here
        results["num_params"] = count_model_params(model)
        all_results[label] = results
    return all_results

# --- Kernel Size sweep ---
kernel_sizes = [1, 3, 5, 7]
configs = [{"kernel_size": int(k)} for k in kernel_sizes]
sweep_results["kernel_size"] = run_sweep_class_hyperparameter(data, SimpleCNN, configs, sweep_name="kernel_size")
plot_sweep(sweep_results["kernel_size"], metric="val_acc", title_prefix="Kernel Size Sweep")
plot_final_comparison(sweep_results["kernel_size"], metric="test_acc", title_prefix="Kernel Size Sweep")

# -- No. Channels sweep ---
channel_list = [i for i in range(5, 25, 5)]
configs = [{"channels": int(c), "linear_in": int(c)} for c in channel_list]
sweep_results["channels"]    = run_sweep_class_hyperparameter(data, SimpleCNN, configs, sweep_name="channels")
plot_sweep(sweep_results["channels"], metric="val_acc", title_prefix="Channels Sweep")
plot_final_comparison(sweep_results["channels"], metric="test_acc", title_prefix="Channels Sweep")

# -- Stride sweep ---
strides = [1, 2, 3, 5]
configs = [{"stride": int(s)} for s in strides]
sweep_results["stride"]      = run_sweep_class_hyperparameter(data, SimpleCNN, configs, sweep_name="stride")
plot_sweep(sweep_results["stride"], metric="val_acc", title_prefix="Stride Sweep")
plot_final_comparison(sweep_results["stride"], metric="test_acc", title_prefix="Stride Sweep")

# %% [markdown]
# ## Summary Table

# %%
summary_df = build_summary_table(sweep_results)
summary_df
