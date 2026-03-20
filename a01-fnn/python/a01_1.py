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
# # Task 1: Implement an MLP
# ## 1a Logistic Regression
#
# The class `LogisticRegression` is located in `a01_functions.py`. You can make
# experimental changes to that class in the other file (`a01_functions.py`). All
# saved changes will be automatically reflected here due to the IPython
# autoreload extension (see below).


# %%
import torch

# %load_ext autoreload
# %autoreload 2
from a01_functions import LogisticRegression, MLP

# %%
# let's test it
# (iii) The model gets initialized with the input dimension D = 3 and output dimension C = 2.
# The input is a random tensor of dimension 3. The output is the log probabilities of the two classes.
# After applying the forward pass, we get the log probabilities of the two classes. By applying the inverse
# operation, the exponential function, we get the actual probailities of the two classes.
logreg = LogisticRegression(3, 2)
x = torch.rand(3)  # input
logreg(x)  # output (log probabilities)
logreg(x).exp()  # output (probabilities)

# %%
# you can access individual parameters as follows
logreg.get_parameter("0_bias")

# %%
# or all of them at once
list(logreg.named_parameters())

# %%
# or directly the tensors stored in the parameters
for par, value in logreg.state_dict().items():
    print(f"{par:<15}= {value}")


# %% [markdown]
# ## 1b MLP
#
# The class `MLP` is also located in `a01_functions.py`. The implementation must
# also be in that file (`a01_functions.py`) as it will be used in later tasks.
#
# Once your implementation is complete, you can proceed with the cells below.

# %%
# here you should see the correct parameter sizes
mlp = MLP([2, 3, 4, 2], torch.relu)
list(mlp.named_parameters())

# %%
# Test your code; we fix the parameters and check the result
with torch.no_grad():
    torch.manual_seed(0)
    for l in range(mlp.num_layers()):
        W, b = mlp.get_parameter(f"{l}_weight"), mlp.get_parameter(f"{l}_bias")
        W[:] = torch.randn(W.shape)
        b[:] = torch.randn(b.shape)

mlp(torch.tensor([-1.0, 2.0]))  # must give: [ 0.8315, -3.6792]

# %%
# Additional by myself: Visualize the MLP
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from a01_helper import saveplot


def draw_mlp(sizes, phi=None, ax=None):
    """Draw a neuron-level MLP diagram for the given layer sizes.

    Parameters
    ----------
    sizes : list of int  – units per layer (input … output)
    phi   : callable     – hidden-layer activation (e.g. torch.relu); None = Linear
    ax    : Axes         – target axes; created if None
    """
    import torch

    if ax is None:
        ax = plt.gca()

    n_layers = len(sizes)
    max_neurons = max(sizes)
    layer_gap = 2.6          # wider gap leaves room for activation boxes
    neuron_radius = 0.18

    # Map common function names to display strings
    _phi_names = {
        "relu": "ReLU", "sigmoid": "Sigmoid", "tanh": "Tanh",
        "silu": "SiLU", "gelu": "GELU", "leaky_relu": "Leaky ReLU", "elu": "ELU",
    }
    if phi is not None:
        raw_name = getattr(phi, "__name__", "").lower()
        phi_name = _phi_names.get(raw_name, raw_name.capitalize() or "φ")
    else:
        phi_name = "Linear"

    # Layer colours and labels
    colours = ["#AED6F1"] + ["#A9DFBF"] * (n_layers - 2) + ["#F9E79F"]
    layer_labels = (
        ["Input"] +
        [f"Hidden {i}" for i in range(1, n_layers - 1)] +
        ["Output"]
    )

    # Neuron centre positions
    centres = []
    for li, n in enumerate(sizes):
        x = li * layer_gap
        y_coords = np.linspace(-(n - 1) / 2, (n - 1) / 2, n)
        centres.append([(x, y) for y in y_coords])

    # ── Edges (drawn first, behind everything) ──────────────────────────────
    for li in range(n_layers - 1):
        for (x0, y0) in centres[li]:
            for (x1, y1) in centres[li + 1]:
                ax.plot([x0, x1], [y0, y1], color="#DDDDDD", lw=0.6, zorder=1)

    # ── Activation mini-plots (between each pair of layers) ─────────────────
    box_hw = 0.36   # half-width  of the activation box (data units)
    box_hh = 0.28   # half-height of the activation box (data units)
    t_np = np.linspace(-2.5, 2.5, 150)

    for li in range(n_layers - 1):
        x_mid = (li + 0.5) * layer_gap
        is_output = (li == n_layers - 2)
        act_label = "Linear" if is_output else phi_name
        act_fn = None if is_output else phi

        # Evaluate activation curve
        if act_fn is not None:
            with torch.no_grad():
                raw_vals = act_fn(
                    torch.tensor(t_np, dtype=torch.float32)
                ).numpy()
        else:
            raw_vals = t_np.copy()

        # Normalise curve to fill the box
        y_range = max(raw_vals.max() - raw_vals.min(), 1e-6)
        y_norm = 2 * (raw_vals - raw_vals.min()) / y_range - 1  # → [-1, 1]
        x_plot = (t_np / t_np.max()) * box_hw + x_mid
        y_plot = y_norm * box_hh

        # Background panel
        rect = plt.Rectangle(
            (x_mid - box_hw - 0.05, -box_hh - 0.05),
            2 * (box_hw + 0.05), 2 * (box_hh + 0.05),
            facecolor="#FFFAF4", ec="#BBBBBB", lw=0.8, ls="--", zorder=2
        )
        ax.add_patch(rect)

        # Zero-axis cross-hairs inside the box
        ax.plot([x_mid - box_hw, x_mid + box_hw], [0, 0],
                color="#CCCCCC", lw=0.5, zorder=3)
        ax.plot([x_mid, x_mid], [-box_hh, box_hh],
                color="#CCCCCC", lw=0.5, zorder=3)

        # Activation curve
        curve_colour = "#AAAAAA" if is_output else "#E74C3C"
        ax.plot(x_plot, y_plot, color=curve_colour, lw=1.8, zorder=4)

        # Label below the box
        ax.text(x_mid, -box_hh - 0.16, act_label,
                ha="center", va="top", fontsize=7.5,
                color=curve_colour, style="italic", fontweight="bold")

    # ── Neurons ─────────────────────────────────────────────────────────────
    for li, layer_centres in enumerate(centres):
        for idx, (x, y) in enumerate(layer_centres):
            circle = plt.Circle(
                (x, y), neuron_radius,
                color=colours[li], ec="#555555", lw=1.2, zorder=5
            )
            ax.add_patch(circle)
            ax.text(x, y, str(idx + 1), ha="center", va="center",
                    fontsize=7, zorder=6)

    # ── Layer labels and unit counts ─────────────────────────────────────────
    for li, (label, n) in enumerate(zip(layer_labels, sizes)):
        x = li * layer_gap
        y_top = (sizes[li] - 1) / 2 + neuron_radius + 0.18
        ax.text(x, y_top, label, ha="center", va="bottom",
                fontsize=9, fontweight="bold")
        ax.text(x, -y_top - 0.1, f"({n} units)", ha="center", va="top",
                fontsize=8, color="#444444")

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-0.6, (n_layers - 1) * layer_gap + 0.6)
    ax.set_ylim(-max_neurons / 2 - 1.0, max_neurons / 2 + 1.0)

    legend_handles = [
        mpatches.Patch(color="#AED6F1", label="Input layer"),
        mpatches.Patch(color="#A9DFBF", label="Hidden layer(s)"),
        mpatches.Patch(color="#F9E79F", label="Output layer"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8)


fig, ax = plt.subplots(figsize=(13, 6))
draw_mlp(mlp.sizes, phi=mlp.phi, ax=ax)
ax.set_title(
    f"MLP Architecture  {mlp.sizes}  |  hidden activation: {mlp.phi.__name__}",
    fontsize=13, fontweight="bold", pad=12
)
plt.tight_layout()
plt.show()

saveplot("a01_1_mlp_architecture.pdf")

# %%
# You can also evaluate your model on multiple inputs at once. Here "torch.func.vmap"
# produces a function that applies the provided function (mlp#forward) to each row of
# its argument (torch.tensor...).
#
# [[ 0.8315, -3.6792],
# [ 4.8448, -6.8813]]
torch.func.vmap(mlp)(torch.tensor([[-1.0, 2.0], [1.0, -2.0]]))

# %% [markdown]
# ## 1c Batching

# %%
# After you adapted the MLP class, you should get the same results as above.
mlp(torch.tensor([-1.0, 2.0]))  # must give: [ 0.8315, -3.6792]

# %%
# Now without vmap. Only proceed to task 2 once this works correctly.
#
# [[ 0.8315, -3.6792],
# [ 4.8448, -6.8813]]
mlp(torch.tensor([[-1.0, 2.0], [1.0, -2.0]]))

# %%
