# ---
# jupyter:
#   jupytext:
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

# %%
import math
import torch
from torch import nn
import torch.nn.functional as F

# %% [markdown]
# # Task 1: Implement an MLP
# ## 1a Logistic Regression


# %%
# `nn.Module` is the superclass of all PyTorch models.
class LogisticRegression(nn.Module):
    """A logistic regression model.

    Parameters
    ----------
    D number of inputs
    C number of classes
    """

    # the definition of all parameters the model uses happens here, i.e., during
    # initialization
    def __init__(self, D, C):
        super().__init__()

        # Create and initialize model parameters. For (multinomial) logistic regression,
        # we have a DxC-dimensional weight matrix W and a C-dimensional bias b.
        self.W = torch.randn(D, C) / math.sqrt(D)
        self.b = torch.randn(C) / math.sqrt(C)

        # Model parameters must be registered to PyTorch as follows. Here we provide
        # a useful name that helps to access/analyze the model later on.
        self.register_parameter("0_weight", nn.Parameter(self.W))
        self.register_parameter("0_bias", nn.Parameter(self.b))

    # the forward function computes the model output for the provided (for this
    # assignment: single) input
    def forward(self, x):
        eta = self.W.t() @ x + self.b
        logprob = F.log_softmax(eta, dim=-1)
        return logprob


# %% [markdown]
# ## 1b MLP


# %%
class MLP(nn.Module):
    """A fully-connected MLP.

    Parameters
    ----------

    sizes Contains the layer sizes. The first entry is the number of inputs, the last
    entry the number of outputs. All entries in between correspond to the number of
    units in the respective hidden layer. E.g., [2,5,7,1] means: 2 inputs -> 5D hidden
    layer -> 7D hidden layer -> 1 output.

    phi Activation function used in every hidden layer (the output layer is linear).

    """

    def __init__(self, sizes: list[int], phi=F.sigmoid):
        super().__init__()

        # let's remember the specification in this model
        self.sizes = sizes
        self.phi = phi

        # Initialize and register the parameters. Follow the naming scheme used for
        # logistic regression above, i.e., the layer-i weights should be named "i_weight" and
        # "i_bias".
        #
        # TODO: YOUR CODE HERE
        for i in range(len(sizes) - 1):
        # Step 1: Extract dimensions from sizes
            D_in, D_out = sizes[i], sizes[i + 1]
        # Step 2: Intialize Tensors and Scale for weights and biases
            W = torch.randn(D_in, D_out) / math.sqrt(D_in)
            b = torch.randn(D_out) / math.sqrt(D_out)
        # Step 4: Register
            self.register_parameter(f"{i}_weight", nn.Parameter(W))
            self.register_parameter(f"{i}_bias", nn.Parameter(b))

    def num_layers(self):
        """Number of layers (excluding input layer)"""
        return len(self.sizes) - 1

    def forward(self, x):
        # TODO: YOUR CODE HERE
        # Step 1: Forward pass through each layer
        for i in range(self.num_layers()):
        # Step 2: Fetch the parameters by using getattr
            W = self.get_parameter(f"{i}_weight")
            b = self.get_parameter(f"{i}_bias")
        # Step 3: Compute the linear transformation 
            #x = W.t() @ x + b
            # Updated to support both 1D vectors and 2D matrices (batches)
            x = x @ W +b
        # Step 4: Apply the activation function (if no activation function, the math will be collapsed -> stacked multiple linear transformations = one big linear transformation= logistic regression)
            if i < self.num_layers() - 1:
                x = self.phi(x)
        # Step 5: Return the output
        return x


# %% [markdown]
# # 2 Multi-Layer Feed-Forward Neural Networks
# ## 2b Train with 2 hidden units

# %%
# Training code. You do not need to modify this code.
from a01_helper import train_scipy, X1, y1

train_bfgs = lambda model, **kwargs: train_scipy(X1, y1, model, **kwargs)


def train1(hidden_sizes, nreps=10, phi=F.sigmoid, train=train_bfgs, **kwargs):
    """Train an FNN.

    hidden_sizes is a (possibly empty) list containing the sizes of the hidden layer(s).
    nreps refers to the number of repetitions.

    """
    best_model = None
    best_cost = math.inf
    for rep in range(nreps):
        model = MLP([1] + hidden_sizes + [1], phi)  # that's your model!
        print(f"X1 shape: {X1.shape}")
        print(f"Repetition {rep: 2d}: ", end="")
        model = train(model, **kwargs)
        mse = F.mse_loss(y1, model(X1)).item()
        if mse < best_cost:
            best_model = model
            best_cost = mse
        print(f"best_cost={best_cost:.3f}")

    return best_model
