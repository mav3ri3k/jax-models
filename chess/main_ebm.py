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
import wandb
from pathlib import Path
from checkpoint import save_checkpoint, restore_checkpoint
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'
print(jax.devices())
import orbax.checkpoint as ocp

from data import prepare_data, load_data
from model import Classifier, EBMChess, EBM
from train import train_step, eval_step, train_step_ebm, eval_step_ebm

# ---------- Paths ----------
# NOTE: use the newly processed tokenized triplet file
data_file = "./data/pre_tokenized/records_cache.arrow"
path = ocp.test_utils.erase_and_create_empty('./checkpoints/')

# ---------- Config ----------
with open("config.toml", "rb") as f:
    cfg = tomllib.load(f)

# run = wandb.init(project="chess", notes="Warmup-32 steps", tags=["ebm"], config=cfg)

# ---------- Model / Optimizer / Metrics ----------
model = EBMChess(cfg, rngs=nnx.Rngs(cfg['seed']))
# jax.debug.visualize_array_sharding(model.energy_model.encoder.blocks[0].ffn.out.kernel.value)
# sys.exit()
lr_schedule = optax.warmup_cosine_decay_schedule(
    init_value= 0.0,
    peak_value=cfg['learning_rate'],
    warmup_steps=cfg['warmup_steps'],
    decay_steps=cfg['total_steps'] - cfg['warmup_steps'],
    end_value=cfg['learning_rate'] / cfg['min_lr_scale'],
)
tx = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(
        learning_rate=lr_schedule,
        b1=cfg['b1'],
        b2=cfg['b2'],
        weight_decay=cfg['weight_decay'],
    ),
)
optimizer = nnx.Optimizer(
    model,
    tx,
)

metrics = nnx.MultiMetric(
    accuracy=nnx.metrics.Accuracy(),
    loss=nnx.metrics.Average('loss'),
)

start_time = time.time()

# ---------- Helper: convert a Polars slice to JAX batch ----------
def df_to_batch(df: pl.DataFrame):
    """
    Expects columns:
      - 'fen_board' : List[int]  (tokenized board; assumed fixed-length per row)
      - 'stk_moves' : List[int]  (tokenized move; assumed single-token -> take idx 0)
    Returns:
      - dict with 'boards': jnp.int32 [B, L], 'moves': jnp.int32 [B]
    """
    # Boards
    boards_list = df.get_column("boards").to_list()  # list[list[int]]
    boards = jnp.asarray(boards_list, dtype=jnp.int32)

    # Moves (take the first token of each list)
    move_lists = df.get_column("moves").to_list()   # list[list[int]]
    try:
        moves_vec = jnp.asarray([mv[0] for mv in move_lists], dtype=jnp.int32)
    except Exception as e:
        # If any row is empty or malformed, raise a clear error
        raise ValueError("Expected each 'stk_moves' row to be a single-token list. "
                         "Found an empty or malformed entry.") from e

    print(boards, moves)
    return {"boards": boards, "moves": moves_vec}

# ---------- Build a small test set once (from the same file) ----------
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

# ---------- Training Loop ----------
with open(data_file, "rb") as f:
    reader = ipc.RecordBatchStreamReader(f)
    batch_size = cfg['batch_size']
    step = 1

    last_checkpoint = cfg['last_checkpoint']
    if last_checkpoint is not False:
        try:
            model = restore_checkpoint(model, f"./checkpoints/{last_checkpoint}", last_checkpoint)
            print(f"[yellow]Start Training from {last_checkpoint}[/yellow]")
        except UnboundLocalError:
            print(f"[red]last_checkpoint: {last_checkpoint} given in config does not exist[/red]")
            sys.exit()
    else:
        print(f"[yellow]Start Training from beginning[/yellow]")

    for batch in reader.iter_batches_with_custom_metadata():  # (RecordBatch)
        # skip ahead if resuming
        if last_checkpoint is not False and step <= last_checkpoint:
            step += 1
            continue

        df = pl.from_arrow(batch[0])  # columns: fen_board, stk_moves, human_moves (all tokenized)
        # chunk into mini-batches
        df_mini = [df[i:i+batch_size] for i in range(0, df.height, batch_size)]

        # shuffle mini-batches per step for SGD
        step_key = jax.random.PRNGKey(step)
        shuffled_indices = jax.random.permutation(step_key, len(df_mini))
        df_mini = [df_mini[int(i)] for i in list(shuffled_indices)]

        # ---- train over this Arrow batch ----
        for df in df_mini:
            stk_moves = df.get_column("moves").to_list()
            stk_moves = jnp.asarray(stk_moves)
            # stk_moves = jnp.squeeze(stk_moves, axis=-1)
            boards = df.get_column("boards").to_list()
            boards = jnp.asarray(boards)

            batch = {"boards": boards, "stk_moves": stk_moves}

            train_step_ebm(model, optimizer, metrics, batch)

        # log train metrics
        m = metrics.compute()
        train_acc = m['accuracy'].item(0)
        train_loss = m['loss'].item(0)
        metrics.reset()

        # ---- eval on fixed small test set ----
        # for df in test_df_mini:
        #     stk_moves = df.get_column("stk_moves").to_list()
        #     stk_moves = jnp.asarray(stk_moves)
            # stk_moves = jnp.squeeze(stk_moves, axis=-1)
        #     boards = df.get_column("fen_board").to_list()
        #     boards = jnp.asarray(boards)

        #     batch = {"boards": boards, "stk_moves": stk_moves}
        #     eval_step_ebm(model, metrics, batch)

        # m = metrics.compute()
        # test_acc = m['accuracy'].item(0)
        # test_loss = m['loss'].item(0)
    
        run.log({
            # "Train_loss": train_loss,
            # "Train_accuracy": train_acc,
            "Test_loss": train_loss,
            "Test_accuracy": train_acc,
        })

        print(f"[green]Finished training:[/green] {step:4d}, "
              f"[cyan]Accuracy:[/cyan] {train_acc:7.4f}, "
              f"[cyan]Loss:[/cyan] {train_loss:7.4f}")
        metrics.reset()

        # checkpoints
        if step in [32, 160, 288, 416, 544, 640] and cfg.get('checkpoint', False):
            save_checkpoint(model, f"./checkpoints/{step}", cfg)

        step += 1

end_time = time.time()
duration = end_time - start_time
run.log({"duration": duration})
run.finish()
# print(f"Training took: {duration:.4f} seconds")

