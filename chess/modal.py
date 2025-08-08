import os
import sys
from rich import print
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import time
import optax
import tomllib
import polars as pl
import pyarrow as pa
import pyarrow.ipc as ipc
import trackio
from pathlib import Path
from rich.progress import track
from rich import print
from checkpoint import save_checkpoint, restore_checkpoint
import modal

import orbax.checkpoint as ocp

from data import prepare_data, load_data
from model import Classifier, EBM
from train import train_step, eval_step, train_step_ebm, eval_step_ebm

app = modal.App("chess")

data_file = "./data/pre_tokenized/cache_tokenized.arrow"
path = ocp.test_utils.erase_and_create_empty('./checkpoints/')

# config
with open("config.toml", "rb") as f:
    cfg = tomllib.load(f)

trackio.init(project="chess-encoder", name="20 EBM Batch", config=cfg)

# model
model = EBM(cfg, rngs=nnx.Rngs(cfg['seed']))
# nnx.display(model)
# sys.exit()
optimizer = nnx.Optimizer(model, optax.adamw(learning_rate=cfg['learning_rate'], b1=cfg['b1'], b2=cfg['b2'], weight_decay=cfg['weight_decay']))


metrics = nnx.MultiMetric(
  accuracy=nnx.metrics.Accuracy(),
  loss=nnx.metrics.Average('loss'),
)

start_time = time.time()

# obtain test set
with open(data_file, "rb") as f:
    reader = ipc.RecordBatchStreamReader(f)
    batch_size = cfg['batch_size']

    test_i = 0
    for batch in reader.iter_batches_with_custom_metadata():
        test_i += 1
        if test_i > 500:
            df = pl.from_arrow(batch[0])
            test_df_mini = [df[i:i+batch_size] for i in range(0, df.height, batch_size)]
            test_i += 1
            break

with open(data_file, "rb") as f:
    reader = ipc.RecordBatchStreamReader(f)

    batch_size = cfg['batch_size']
    step = 1

    last_checkpoint = cfg['last_checkpoint']
    # restore checkpoint from r2 bucket
    if last_checkpoint != False:
        try:
            model = restore_checkpoint(model, f"./checkpoints/{last_checkpoint}", last_checkpoint)
            print(f"[yellow]Start Training from {last_checkpoint}[/yellow]")
        except UnboundLocalError:
            print(f"[red]Last_checkpoint: {last_checkpoint} given in config does not exist[/red]")
            sys.exit()
    else:
        print(f"[yellow]Start Training from beginning[/yellow]")

    for batch in reader.iter_batches_with_custom_metadata():
        if last_checkpoint != False:
            while step <= last_checkpoint:
                step += 1
                continue

        df = pl.from_arrow(batch[0])

        df_mini = [df[i:i+batch_size] for i in range(0, df.height, batch_size)]

        # shuffle df_mini
        step_key = jax.random.PRNGKey(step)
        shuffled_indices = jax.random.permutation(step_key, len(df_mini))
        df_mini = [df_mini[i] for i in shuffled_indices]

        for df in df_mini:
            moves = df.get_column("moves").to_jax()
            boards = df.get_column("boards").to_list()
            boards = jnp.asarray(boards)

            batch = {"boards": boards, "moves": moves}

            train_step_ebm(model, optimizer, metrics, batch)

        m = metrics.compute()
        train_acc = m['accuracy'].item(0)
        train_loss = m['loss'].item(0)
        metrics.reset()

        # eval
        for df in test_df_mini:
            moves = df.get_column("moves").to_jax()
            boards = df.get_column("boards").to_list()
            boards = jnp.asarray(boards)

            batch = {"boards": boards, "moves": moves}

            eval_step_ebm(model, metrics, batch)

        m = metrics.compute()
        test_acc = m['accuracy'].item(0)
        test_loss = m['loss'].item(0)
        trackio.log({
            "Train_loss": train_loss,
            "Train_accuracy": train_acc,
            "Test_loss": test_loss,
            "Test_accuracy": test_acc,
        })

        print(f"[green]Finished training:[/green] {step:4d}, [cyan]Accuracy:[/cyan] {test_acc:7.4f}, [cyan]Loss:[/cyan] {test_loss:7.4f}")
        metrics.reset()

        if step in [32, 160, 288, 416, 544, 640] and cfg['checkpoint']:
            save_checkpoint(model, f"./checkpoints/{step}", cfg)
        step += 1

end_time = time.time()
duration = end_time - start_time
# print(f"  Training took: {duration:.4f} seconds")

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
