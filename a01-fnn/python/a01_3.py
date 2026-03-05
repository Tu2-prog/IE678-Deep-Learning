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

# %%
import torch
import torch.nn.functional as F

# import helper functions
import sys, os

sys.path.append(os.getcwd())
from a01_helper import *

# %load_ext autoreload
# %autoreload 2
from a01_functions import train1


# %% [markdown]
# # 3 Backpropagation

# %%
# Let's fit the model with one hidden layer consisting of 50 units.
model = train1([50], nreps=1)
print("Training error:", F.mse_loss(y1, model(X1)).item())
print("Test error    :", F.mse_loss(y1test, model(X1test)).item())

# Extract parameters
pars = dict(model.named_parameters())
W1 = pars["0_weight"].data  # 1x50
b1 = pars["0_bias"].data  # 50
W2 = pars["1_weight"].data  # 50x1
b2 = pars["1_bias"].data  # 1

# %% [markdown]
# ## 3a Forward pass

# %%
# Compute results of forward pass on an example x (i.e., z1, z2, z3, z4, yhat, l) using Pytorch
x = X1test[1, :]
y = y1test[1, :]
print(f"x={x}, y={y}, yhat={model(x).detach()}, l={torch.nn.MSELoss()(y, model(x))}")

# %%
# Now do this by hand (including all intermediate values). You should get the same
# results as above.

z1 = x @ W1
z2 = z1 + b1
z3 = F.sigmoid(z2)
z4 = z3 @ W2
yhat = z4 + b2
l = torch.pow(y - yhat, 2)
print(f"x={x}, y={y}, yhat={model(x).detach()}, l={torch.nn.MSELoss()(y, model(x))}")

# %% [markdown]
# ## 3b Backward pass

# %%
# Compute results of backward pass on example output (i.e., delta_x, delta_W1, delta_z1,
# delta_b1, delta_z2, delta_z3, delta_W2, delta_z4, delta_b2, delta_yhat, delta_l, delta_y)

delta_l = 1
delta_yhat = -2 * (y - yhat) * delta_l
delta_y = 2 * (y - yhat) * delta_l

delta_b2 = delta_yhat
delta_z4 = delta_yhat

delta_W2 = torch.outer(z3, delta_z4)
delta_z3 = delta_z4 @ W2.T

delta_z2 = delta_z3 * z3 * (1 - z3)
delta_b1 = delta_z2

delta_z1 = delta_z2
delta_W1 = torch.outer(x, delta_z1)
delta_x  = delta_z1 @ W1.T
# %%
# Use PyTorch's backprop
x.requires_grad = True
y.requires_grad = True
if x.grad is not None:
    x.grad.zero_()
if y.grad is not None:
    y.grad.zero_()
model.zero_grad()
t_yhat = model(x)
t_yhat.retain_grad()
t_l = torch.nn.MSELoss()(t_yhat, y)
t_l.backward()
t_delta_l = 1
t_delta_y = y.grad
t_delta_yhat = t_yhat.grad
t_delta_b2 = model.get_parameter("1_bias").grad
t_delta_W2 = model.get_parameter("1_weight").grad
t_delta_b1 = model.get_parameter("0_bias").grad
t_delta_W1 = model.get_parameter("0_weight").grad
t_delta_x = x.grad

# %%
# Check if equal (show squared error)
for v in ["y", "yhat", "b2", "W2", "b1", "W1", "x"]:
    pyt = eval("t_delta_" + v)
    you = torch.tensor(eval("delta_" + v))
    print(f"{v}, squared error={torch.sum((pyt - you) ** 2)}")

# %%
# Check if equal (show actual values)
for v in ["l", "y", "yhat", "b2", "W2", "b1", "W1", "x"]:
    pyt = eval("t_delta_" + v)
    you = torch.tensor(eval("delta_" + v))
    print(f"{v}, pytorch={pyt}, you={you}")
