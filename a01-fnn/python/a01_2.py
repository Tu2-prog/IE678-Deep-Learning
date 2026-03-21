# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     custom_cell_magics: kql
#     formats: py:percent,ipynb
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
# # DL26, Assignment 1
# **Anh Tu Duong Nguyen** (anguyea, 2115931)
#
# **Anh-Nhat Nguyen** (anhnnguy, 2034311)

# %%
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# import helper functions
import sys, os

sys.path.append(os.getcwd())
# %load_ext autoreload
# %autoreload 2
from a01_helper import *
from a01_functions import train1, MLP  # MLP is implicitly required for `torch.load`

# %% [markdown]
# # 2 Multi-Layer Feed-Forward Neural Networks
# ## 2a Conjecture how an FNN fit will look like

# %%
# here is the one-dimensional dataset that we will use
nextplot()
plot1(X1, y1, label="train")
plot1(X1test, y1test, label="test")
plt.legend(loc="upper right")

# Use this function to save your plot.
saveplot("a01_2_data.pdf")

# %% [markdown]
# ## 2b Train with 2 hidden units

# %%
# Let's fit the model with one hidden layer consisting of 2 units.
model = train1([2], nreps=1)
print("Training error:", F.mse_loss(y1, model(X1)).item())
print("Test error    :", F.mse_loss(y1test, model(X1test)).item())

# %%
# plot the data and the fit
nextplot()
plot1(X1, y1, label="train")
plot1(X1test, y1test, label="test")
plot1fit(torch.linspace(0, 13, 500).unsqueeze(1), model)

saveplot("a01_2_fit-2-neurons.pdf")

# %%
# The weight matrices and bias vectors can be read out as follows. If you want, use
# these parameters to compute the output of the network (on X1) directly and compare to
# vmap(model)(X1).
for par, value in model.state_dict().items():
    print(f"{par:<15}= {value}")

# %%
# now repeat this multiple times
# TODO: YOUR CODE HERE
nextplot()

plot1(X1, y1, label="train")
plot1(X1test, y1test, label="test")

X_plot = torch.linspace(0, 13, 500).unsqueeze(1)
colors = plt.cm.tab10.colors   # 10 distinct colours from matplotlib's default cycle
for i in range(5):
    print(f"\n--- Run {i+1} ---")
    temp_model = train1([2], nreps=1)
    lines = plt.plot(
        X_plot.numpy(),
        temp_model(X_plot).detach().numpy(),
        label=f"Run {i+1}",
        color=colors[i + 2],   # offset by 2 to avoid reusing train/test colours
        linewidth=1.4,
        alpha=0.85,
    )

plt.title("5 independent fits  –  MLP([1, 2, 1])", fontsize=12)
plt.xlabel("x")
plt.ylabel("y")
plt.legend(
    loc="upper left",
    bbox_to_anchor=(1.01, 1),   # place legend outside the axes on the right
    borderaxespad=0,
    fontsize=9,
)
plt.tight_layout()
saveplot("a01_2_repeated-fits.pdf")
# %% [markdown]
# ## 2c Width

# %%
# Experiment with different hidden layer sizes. To avoid recomputing
# models, you may want to save your models using torch.save(model, filename) and
# load them again using torch.load(filename).
# TODO: YOUR CODE HERE
hidden_sizes_list = [1, 2, 3, 10, 50, 100]
models = {}
train_mses = {}
test_mses = {}

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

for h in hidden_sizes_list:
    fname = os.path.join(MODEL_DIR, f"a01_2c_model_h{h}.pt")
    if os.path.exists(fname):
        print(f"=== Loading cached model  h={h} from {fname} ===")
        model_h = torch.load(fname, weights_only=False)
    else:
        print(f"\n=== Training MLP  h={h} ===")
        model_h = train1([h], nreps=10)
        torch.save(model_h, fname)
        print(f"    Saved to {fname}")
    models[h] = model_h

    train_mse = F.mse_loss(y1, model_h(X1)).item()
    test_mse  = F.mse_loss(y1test, model_h(X1test)).item()
    train_mses[h] = train_mse
    test_mses[h]  = test_mse
    print(f"  Train MSE: {train_mse:.6f}  |  Test MSE: {test_mse:.6f}")

# Print summary table
print("\n" + "=" * 50)
print(f"{'Hidden Units':<15} {'Train MSE':<15} {'Test MSE':<15}")
print("=" * 50)
for h in hidden_sizes_list:
    print(f"{h:<15} {train_mses[h]:<15.6f} {test_mses[h]:<15.6f}")
print("=" * 50)

# Plot all predictions in a single plot
nextplot()
plot1(X1, y1, label="train")
plot1(X1test, y1test, label="test")

X_plot = torch.linspace(0, 13, 500).unsqueeze(1)
colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(hidden_sizes_list)))

for i, h in enumerate(hidden_sizes_list):
    y_pred = models[h](X_plot).detach().numpy()
    plt.plot(
        X_plot.numpy(),
        y_pred,
        label=f"h={h} (test MSE={test_mses[h]:.3f})",
        color=colors[i],
        linewidth=1.5,
        alpha=0.85,
    )

plt.title("FNN predictions with varying hidden layer sizes", fontsize=12)
plt.xlabel("x")
plt.ylabel("y")
plt.ylim(-1.5, 1.5)   # clamp y-axis to the data range; large-h spikes would otherwise squash the view
plt.legend(
    loc="upper left",
    bbox_to_anchor=(1.01, 1),
    borderaxespad=0,
    fontsize=8,
)
plt.tight_layout()
saveplot("a01_2_width-comparison.pdf")

# %% [markdown]
# ## 2d Distributed representations

# %%
# train a model to analyze
model = train1([2])

# TODO: YOUR CODE HERE

# %%
# plot the fit as well as the outputs of each neuron in the hidden
# layer (scale for the latter is shown on right y-axis)
nextplot()
plot1(X1, y1, label="train")
plot1(X1test, y1test, label="test")
plot1fit(torch.linspace(0, 13, 500).unsqueeze(1), model, hidden=True, scale=False)
saveplot("a01_2_distributed-reps.pdf")

# %%
# plot the fit as well as the outputs of each neuron in the hidden layer, scaled
# by its weight for the output neuron (scale for the latter is shown on right
# y-axis)
nextplot()
plot1(X1, y1, label="train")
plot1(X1test, y1test, label="test")
plot1fit(torch.linspace(0, 13, 500).unsqueeze(1), model, hidden=True, scale=True)
plt.legend(loc="upper right")
plt.tight_layout()
saveplot("a01_2_distributed-reps-scaled.pdf")

# %%
# 2d(iii): Repeat (i) and (ii) with 3 hidden neurons
# TODO: YOUR CODE HERE
X_dense = torch.linspace(0, 13, 500).unsqueeze(1)

model_h3 = train1([3])

nextplot()
plot1(X1, y1, label="train")
plot1(X1test, y1test, label="test")
plot1fit(X_dense, model_h3, hidden=True, scale=False)
saveplot("a01_2_distributed-reps-h3.pdf")

nextplot()
plot1(X1, y1, label="train")
plot1(X1test, y1test, label="test")
plot1fit(X_dense, model_h3, hidden=True, scale=True)
plt.legend(loc="upper right")
plt.tight_layout()
saveplot("a01_2_distributed-reps-h3-scaled.pdf")

# %%
# 2d(iii): Repeat (i) and (ii) with 10 hidden neurons
model_h10 = train1([10])

nextplot()
plot1(X1, y1, label="train")
plot1(X1test, y1test, label="test")
plot1fit(X_dense, model_h10, hidden=True, scale=False)
saveplot("a01_2_distributed-reps-h10.pdf")

nextplot()
plot1(X1, y1, label="train")
plot1(X1test, y1test, label="test")
plot1fit(X_dense, model_h10, hidden=True, scale=True)
plt.legend(loc="upper right")
plt.tight_layout()
saveplot("a01_2_distributed-reps-h10-scaled.pdf")

# %% [markdown]
# ## 2e Experiment with different optimizers (optional)

# %%
# PyTorch provides many gradient-based optimizers; see
# https://pytorch.org/docs/stable/optim.html. You can use a PyTorch optimizer
# as follows.
train_adam = lambda model, **kwargs: fnn_train(
    X1, y1, model, optimizer=torch.optim.Adam(model.parameters(), lr=0.01), **kwargs
)
model = train1([50], nreps=1, train=train_adam, max_epochs=5000, tol=1e-8, verbose=True)

# %%
# Experiment with different number of layers and activation functions. Here is
# an example with three hidden layers (of sizes 4, 5, and 6) and ReLU activations.
#
# You can also plot the outputs of the hidden neurons in the first layer (using
# the same code above).
model = train1([4, 5, 6], nreps=50, phi=F.relu)
nextplot()
plot1(X1, y1, label="train")
plot1(X1test, y1test, label="test")
plot1fit(torch.linspace(0, 13, 500).unsqueeze(1), model)
print("Training error:", F.mse_loss(y1, model(X1)).item())
print("Test error    :", F.mse_loss(y1test, model(X1test)).item())
