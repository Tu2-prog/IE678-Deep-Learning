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
# Anh Tu Duong Nguyen (anguyea, 115931)
#
# Anh-Nhat Nguyen (anhnnguy, 2034311)

# %%
import torch
import numpy as np

# %load_ext autoreload
# %autoreload 2
from a02_functions import ClimbCNN


# %% [markdown]
#
# # Task 1: Climbing with a CNN

# %% [markdown]
# Initialize `model` using the right hyperparameters for task 1a). Then inspect your
# model's parameters using `model.state_dict()`. (Note that when using layer
# implementations from `torch.nn`, parameters are initialized randomly as discussed in
# the lecture.)

# %%
# TODO: your code here
model = ClimbCNN(in_channels=1, out_channels=1, kernel_size=2)
model.state_dict()

# %% [markdown]
# You can access the model parameters via `<model>.<param-name>`. Set all parameters to
# the values required to solve task a).

# %%
with torch.no_grad():
        # TODO: your code here
    model.conv.weight.copy_(torch.tensor([[[-1.0, 1.0]]]))  # difference kernel to compute increment as we are sliding from left to right -> left value needs to be subtracted from the right value
    model.conv.bias.copy_(torch.tensor([0.0]))

# %% [markdown]
# Simple test case that can be verified by hand.

# %%
x = torch.Tensor([0.0, 5.0, 11.0, 7.0, 15.0, 3.0]).view(1, -1)
with torch.no_grad():
    y = model(x)
print(y)  # should give: tensor([19.])

# %% [markdown]
# Example from the lecture.

# %%
climb_data = (
    torch.from_numpy(np.genfromtxt("climb_filtered.csv")).to(torch.float).view(1, -1)
)
with torch.no_grad():
    y = model(torch.Tensor(climb_data).view(1, -1))

print(y)  # should give: tensor([503.4000])

# %% [markdown]
# Now create a new `model2` that computes that total ascent (first output) and total descent (second output) simultaneously. Do this using the same model implementation as above, but change only hyperparameters and parameters.

# %%
# TODO: your code here
model2 = ClimbCNN(in_channels=1, out_channels=2, kernel_size=2)

with torch.no_grad():
    model2.conv.weight.copy_(torch.tensor([
        [[-1.0,  1.0]],  # ascent kernel matrix the same as before
        [[ 1.0, -1.0]],  # descent kernel matrix is the flipped kernel as we are computing the opposite of the ascent
    ]))
    model2.conv.bias.copy_(torch.tensor([0.0, 0.0]))  # one bias per output channel

# %%
x = torch.Tensor([0.0, 5.0, 11.0, 7.0, 15.0, 3.0]).view(1, -1)
with torch.no_grad():
    y = model2(x)
print(y)  # should give: [19., 16.]

# %%
with torch.no_grad():
    y = model2(torch.Tensor(climb_data).view(1, -1))

print(y)  # should give: tensor([503.4000, 513.6000])
