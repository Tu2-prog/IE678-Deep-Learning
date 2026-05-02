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
#     display_name: dl-2
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import torch
from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from torch import nn
from a03_helper import (
    nextplot,
    tsne_vocab,
    tsne_thought,
    DEVICE,
    reviews_load_embeddings,
)
from a03_functions import SimpleLSTM, ReviewsDataset, ReviewsDataModule, LitSimpleLSTM
import matplotlib.pyplot as plt

# %% [markdown]
# ## Task 5: Pre-trained Embeddings & Visualization

# %% [markdown]
# ### Task 5b

# %%
# Load Glove embeddings into a plain embedding layer.
dataset = ReviewsDataset(use_vocab=True)
vocab = dataset.vocab
glove_embeddings = nn.Embedding(len(vocab), 100, device=DEVICE)
reviews_load_embeddings(glove_embeddings, vocab.get_stoi())

# %%
# Print one embedding
glove_embeddings(torch.tensor(vocab["movie"], device=DEVICE))

# %%
# Plot embeddings of first 100 words using t-SNE
nextplot()
_ = tsne_vocab(glove_embeddings, torch.arange(100), vocab)

# %%
# You can also specify colors and/or drop the item labels
nextplot()
_ = tsne_vocab(glove_embeddings, torch.arange(100), colors=[0] * 50 + [1] * 50)

# %%
# YOUR CODE HERE
# Note: you can obtain the embeddings tensor using glove_embeddings.weight.data
import torch.nn.functional as F

# Get the actual first 100 words from vocab
n = 100
words_n = [vocab.lookup_token(i) for i in range(n)]
indices_n = torch.arange(n, device=DEVICE)
selected = glove_embeddings.weight[indices_n]

# (i) t-SNE of first 100 words
nextplot()
_ = tsne_vocab(glove_embeddings, indices_n, vocab, colors=[0] * 50 + [1] * 50)

# %%
# (ii) Cosine similarity heatmap for first 100 words
sim_matrix = F.cosine_similarity(selected.unsqueeze(1), selected.unsqueeze(0), dim=2)
plt.figure(figsize=(20, 20))
plt.imshow(sim_matrix.detach().cpu().numpy(), cmap="coolwarm", vmin=-1, vmax=1)
plt.xticks(range(100), words_n, rotation=90, fontsize=6)
plt.yticks(range(100), words_n, fontsize=6)
plt.colorbar(label="Cosine Similarity")
plt.title("Cosine Similarity Matrix of First N Vocab Words")
plt.tight_layout()

# %% [markdown]
# ### Task 5c

# %%
# hyperparameter settings for rest of task 5
vocab_size = len(dataset.vocab)
embedding_dim = 100
hidden_dim = 100
num_layers = 2
n_epochs = 10
cell_dropout = 0.0

# %%
model = LitSimpleLSTM(vocab_size, embedding_dim, hidden_dim, num_layers, cell_dropout)
dataset = ReviewsDataset(use_vocab=True)
dm = ReviewsDataModule(dataset)
# TODO: Your code here
# Train a plain model so that it reaches a train accuracy of >0.9.
trainer = Trainer(max_epochs=10, check_val_every_n_epoch=1, logger=TensorBoardLogger("tb_logs", name="a03-rnn"))
trainer.fit(model, datamodule=dm)

trainer.test(model, datamodule=dm)

# %%
# Plot t-SNE embeddings of the thought vectors for training data
# point color = label
dm.setup("fit")
train_loader = dm.train_dataloader()
nextplot()
_ = tsne_thought(model, train_loader, DEVICE)

# %%
# Plot t-SNE embeddings of of the thought vectors for validation data
dm.setup("fit")
val_loader = dm.val_dataloader()
nextplot()
_ = tsne_thought(model, val_loader, DEVICE)

# %% [markdown]
# ### Task 5d

# %%
# Initialize the model with *p*re-trained embeddings with *f*inetuning, then
# train.
model_pf = LitSimpleLSTM(
    vocab_size, embedding_dim, hidden_dim, num_layers, cell_dropout
)
reviews_load_embeddings(model_pf.model.embedding, vocab.get_stoi())
# TODO: Your code here

# Train with GloVe-initialized embeddings
trainer_pf = Trainer(max_epochs=10, gradient_clip_val=3, check_val_every_n_epoch=1, logger=TensorBoardLogger("tb_logs", name="a03-rnn-glove"))
trainer_pf.fit(model_pf, datamodule=dm)

trainer_pf.test(model_pf, datamodule=dm)

# (i) Training set thought vectors
dm.setup("fit")
train_loader = dm.train_dataloader()
nextplot()
_ = tsne_thought(model_pf, train_loader, DEVICE)

# (ii) Validation set thought vectors
val_loader = dm.val_dataloader()
nextplot()
_ = tsne_thought(model_pf, val_loader, DEVICE)

# %% [markdown]
# ### Task 5e

# %%
# Initialize the model with *p*re-trained embeddings without finetuning, then
# train.
model_p = LitSimpleLSTM(vocab_size, embedding_dim, hidden_dim, num_layers, cell_dropout)
reviews_load_embeddings(model_p.model.embedding, vocab.get_stoi())
model_p.model.embedding.weight.requires_grad = False
# TODO: Your code here

trainer_p = Trainer(max_epochs=10, gradient_clip_val=3, check_val_every_n_epoch=1, logger=TensorBoardLogger("tb_logs", name="a03-rnn-glove"))
trainer_p.fit(model_p, datamodule=dm)

trainer_p.test(model_p, datamodule=dm)

# (i) Training set thought vectors
dm.setup("fit")
train_loader = dm.train_dataloader()
nextplot()
_ = tsne_thought(model_p, train_loader, DEVICE)

# (ii) Validation set thought vectors
val_loader = dm.val_dataloader()
nextplot()
_ = tsne_thought(model_p, val_loader, DEVICE)

# %% [markdown]
# # 5f)

# %%
from lightning.pytorch import Trainer
from lightning.pytorch.tuner import Tuner

datamodule = ReviewsDataModule(dataset)
datamodule.setup("fit")

# Find optimal learning rate
model_tuning = LitSimpleLSTM(vocab_size, embedding_dim, hidden_dim, num_layers, cell_dropout)
datamodule = ReviewsDataModule(dataset)
datamodule.setup("fit")

trainer = Trainer(max_epochs=10)
tuner = Tuner(trainer)

# Automatically finds the best lr
lr_finder = tuner.lr_find(model_tuning, datamodule=datamodule)
best_lr = lr_finder.suggestion()
print(f"Best lr: {best_lr}")

# Train with the best lr
model_tuning.lr = best_lr
trainer = Trainer(max_epochs=10)
trainer.fit(model_tuning, datamodule=datamodule)
trainer.test(model_tuning, datamodule=datamodule)

# %%
# (i) Training set thought vectors
dm.setup("fit")
train_loader = dm.train_dataloader()
nextplot()
_ = tsne_thought(model_tuning, train_loader, DEVICE)

# (ii) Validation set thought vectors
val_loader = dm.val_dataloader()
nextplot()
_ = tsne_thought(model_tuning, val_loader, DEVICE)