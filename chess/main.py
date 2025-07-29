import os
import sys
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import time
import optax
import tomllib
import polars as pl
import pyarrow as pa
import pyarrow.ipc as ipc
from pathlib import Path
from tqdm import tqdm, trange

import orbax.checkpoint as ocp
from orbax.checkpoint import CheckpointManager, CheckpointManagerOptions, args

from data import prepare_data, load_data
from model import VisionTransformer
from train import train_step, eval_step

data_file = "./data/pre_tokenized/cache_tokenized.arrow"

# config
with open("config.toml", "rb") as f:
    cfg = tomllib.load(f)

# model
model = VisionTransformer(cfg, rngs=nnx.Rngs(cfg['seed']))
optimizer = nnx.Optimizer(model, optax.adamw(cfg['learning_rate']))

metrics = nnx.MultiMetric(
  accuracy=nnx.metrics.Accuracy(),
  loss=nnx.metrics.Average('loss'),
)

print("TRAIN")
for epoch in trange(cfg["epochs"]):
    print(f"Epoch {epoch+1}")

    start_time = time.time()

    with open(data_file, "rb") as f:
        reader = ipc.RecordBatchStreamReader(f)

        total_record_batches = 10
        # for  _ in reader.iter_batches_with_custom_metadata():
        #     total_record_batches += 1

        # time an epoch
        j = 1
        for batch in tqdm(reader.iter_batches_with_custom_metadata(), total=total_record_batches):
            batch_size = 50
            df = pl.from_arrow(batch[0])

            # assert df.height % batch_size == 0, f"{df.height}, {batch_size}"

            df_mini = [df[i:i+batch_size] for i in range(0, df.height, batch_size)]
            for df in df_mini:
                moves = df.get_column("moves").to_jax()
                boards = df.get_column("boards").to_list()
                boards = jnp.asarray(boards)

                batch = {"boards": boards, "moves": moves}

                train_step(model, optimizer, metrics, batch)

            j += 1
            if j > total_record_batches:
                break

    print(metrics)
    m = metrics.compute()
    print(f"  Accuracy: {m['accuracy']}, Loss: {m['loss']}" )
    metrics.reset()
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"  Training took: {duration:.4f} seconds")

# test
"""
print("TEST")
test_metrics = nnx.MultiMetric(
            accuracy=nnx.metrics.Accuracy(),
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
print(f"  Accuracy: {m['accuracy']}, Loss: {m['loss']}" )
metrics.reset()
"""
