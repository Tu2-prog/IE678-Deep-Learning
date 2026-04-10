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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# Anh Tu Duong Nguyen (anguyea, 115931)
#
# Anh-Nhat Nguyen (anhnnguy, 2034311)

# %%
import pandas as pd
import torch
from itertools import product

from a02_functions import SimpleCNN, train_model
from a02_helper import get_raw_data, count_model_params

# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# # Task 4: Exploration

# %%
# TODO: your code here
