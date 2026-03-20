# ---
# jupyter:
#   jupytext:
#     custom_cell_magics: kql
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
import math
import torch
from torch import nn
import torch.nn.functional as F

# %% [markdown]
# # Task 1: Implement an MLP
# ## 1a Logistic Regression


# %%
# `nn.Module` is the superclass of all PyTorch models.
# 1.a)
# (i) The weights and biases are initialized with standard tensors of dimensions (D, C) and (C) respectively.
# The tensors are initialized with random values and then scaled by the square root of D.
# The parameters are formally stored using the self.register_parameter method. By doing so the parameters are assigned identifiers that
# allows PyTorch to keep track of them to optimize them during training.

# (ii) The forward pass is a simple linear transformation of the input x using the weight matrix W.
# This linear transformation gets the softmax function applied to it to get the log probabilities of the classes as outputs.
# The softmax function works as an activation function that converts the logits to probabilities which is considered standard practice.
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
        # Computation is down iteratively for every layer. IMPORTANT: We are not iterating over a tensor here, but just over the number of layers.
        for i in range(self.num_layers()):
            # The parameters are registerd with unique identifiers. We can access them to get the parameters of our corresponding layer.
            W = self.get_parameter(f"{i}_weight")
            b = self.get_parameter(f"{i}_bias")

            # The output of the layer is the linear transformation of the input x using the weight matrix of the layer with a linear activation function, but only if the current layer is not the output layer.
            # To support both single input vectors and batches of input vectors, we are simply swapping x and W so the inner dimensions match during this matrix multiplication without the need of using a for loop.
            # x = x @ W + b <- old method 
            x = x @ W + b
            if i < self.num_layers() - 1:
                x = self.phi(x)
        # If input was a vector, remove batch dimension from output
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
