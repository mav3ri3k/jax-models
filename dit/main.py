import os
import sys
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import time
import optax
import tomllib
from pathlib import Path
from tqdm import tqdm, trange

import orbax.checkpoint as ocp
from orbax.checkpoint import CheckpointManager, CheckpointManagerOptions, args

from data import prepare_data, load_data
from model import VisionTransformer
from train import train_step, eval_step

# data
if not Path("cifar10_uint8.npz").exists():
    prepare_data()

data = load_data()
train_images = jnp.asarray(data["train_images"])  # shape (50000,32,32,3), dtype uint8
train_labels = jnp.asarray(data["train_labels"])  # shape (50000,), dtype int64
test_images  = jnp.asarray(data["test_images"])   # shape (10000,32,32,3)
test_labels  = jnp.asarray(data["test_labels"])   # shape (10000,)

# config
with open("config.toml", "rb") as f:
    cfg = tomllib.load(f)

# model
model = VisionTransformer(cfg, rngs=nnx.Rngs(cfg['seed']))
optimizer = nnx.Optimizer(model, optax.adamw(cfg['learning_rate']))

metrics = nnx.MultiMetric(
  loss=nnx.metrics.Average('loss'),
)
print("TRAIN")
for epoch in trange(cfg["epochs"]):
    print(f"Epoch {epoch+1}")

    num_train = train_images.shape[0]
    batch_size = cfg['batch_size']

    perm = jax.random.permutation(jax.random.PRNGKey(epoch), num_train)
    shuffled_imgs = train_images[perm]
    shuffled_lbls = train_labels[perm]

    batches = [
        {"images": shuffled_imgs[i : i + batch_size],
         "labels": shuffled_lbls[i : i + batch_size],
         "keys"  : epoch + i + jnp.arange(batch_size),
         "times" :  jax.random.randint(key=jax.random.PRNGKey(epoch + i), shape=(batch_size,), minval=0, maxval=cfg['diff_steps']),
         "eps"   : jnp.asarray([jax.random.normal(key=jax.random.PRNGKey(epoch + i), shape=shuffled_imgs[0].shape) for i in range(batch_size)])}
        for i in range(0, num_train, batch_size)
    ]

    # time an epoch
    start_time = time.time()
    for batch in tqdm(batches):
        train_step(model, optimizer, metrics, batch)

    m = metrics.compute()
    print(f"  Loss: {m['loss']}" )
    metrics.reset()
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"  Training took: {duration:.4f} seconds")

# test
print("TEST")
test_metrics = nnx.MultiMetric(
            loss=nnx.metrics.Average('loss'),
        )

num_test = test_images.shape[0]
for i in range(0, num_test, cfg['batch_size']):
    batch = {
        "images": test_images[i : i + cfg['batch_size']],
        "labels": test_labels[i : i + cfg['batch_size']],
    }
    eval_step(model, test_metrics, batch)

m = test_metrics.compute()
print(f"  Loss: {m['loss']}" )
metrics.reset()
