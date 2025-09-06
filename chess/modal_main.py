import modal
from pathlib import Path, PosixPath
import tomllib

MINUTES = 60  # seconds
HOURS = 60 * MINUTES

app_name = "chess"
app = modal.App(app_name)
gpu = "L40S"
volume = modal.Volume.from_name("chess_data", create_if_missing=False)
volume_path = PosixPath("/vol")
volume_data_path = volume_path / "data"
chk_path = volume_path / "checkpoint"
model_save_path = volume_path / "models"
data_file_path = volume_data_path / "records_cache.arrow"

image = (
    modal.Image.debian_slim(python_version="3.11")
    # .uv_pip_install("torch==2.6.0")
    .uv_pip_install("jax[cuda12]")
    .uv_pip_install("boto3>=1.40.1")
    .uv_pip_install("dotenv>=0.9.9")
    .uv_pip_install("flax==0.10.4")
    .uv_pip_install("optax>=0.2.5")
    .uv_pip_install("orbax-checkpoint>=0.11.5")
    .uv_pip_install("polars>=1.31.0")
    .uv_pip_install("pyarrow>=20.0.0")
    .uv_pip_install("rich>=14.0.0")
    .uv_pip_install("wandb>=0.21.3")
    .add_local_python_source("train")
    .add_local_python_source("model")
    .add_local_python_source("checkpoint")
    .add_local_python_source("upload")
    # .run_commands("git clone https://github.com/modal-labs/agi && echo 'ready to go!'")
)

with image.imports():
    import os
    import sys
    from rich import print
    import jax
    import jax.numpy as jnp
    import flax.nnx as nnx
    import time
    import optax
    import polars as pl
    import pyarrow as pa
    import pyarrow.ipc as ipc
    import wandb
    from pathlib import Path
    from checkpoint import save_checkpoint_modal, restore_checkpoint_modal

    import orbax.checkpoint as ocp

    from model import Classifier, EBMChess, EBM
    from train import train_step, eval_step, train_step_ebm, eval_step_ebm

@app.function(
    image=image,
    volumes={volume_path: volume},
    gpu=gpu,
    timeout=30 * MINUTES,
    secrets=[modal.Secret.from_name("wandb-secret")]
)
def train_model(cfg, data_file_path, chk_path):
    data_file = data_file_path
    # path = ocp.test_utils.erase_and_create_empty('./checkpoints/')

    run = wandb.init(project="chess", name="4098", notes="4098 steps", tags=[cfg['type']], config=cfg)

    # ---------- Model / Optimizer / Metrics ----------
    if cfg['type'] == "cls":
        model = Classifier(cfg, rngs=nnx.Rngs(cfg['seed']))
    else:
        model = EBMChess(cfg, rngs=nnx.Rngs(cfg['seed']))

    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value= 0.0,
        peak_value=cfg['learning_rate'],
        warmup_steps=cfg['warmup_steps'],
        decay_steps=cfg['total_steps'],
        end_value=0,
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
                model = restore_checkpoint_modal(model, chk_path / cfg['type'] / cfg['size'] / f"{last_checkpoint}", last_checkpoint)
                print(f"[yellow]Start Training from {last_checkpoint}[/yellow]")
            except UnboundLocalError:
                print(f"[red]last_checkpoint: {last_checkpoint} given in config does not exist[/red]")
                sys.exit()
        else:
            print(f"[yellow]Start Training from beginning[/yellow]")


        for batch in reader.iter_batches_with_custom_metadata():  # (RecordBatch)
            df = pl.from_arrow(batch[0])  # columns: fen_board, stk_moves, human_moves (all tokenized)
            # chunk into mini-batches
            df_mini = [df[i:i+batch_size] for i in range(0, df.height, batch_size)]

            # shuffle mini-batches per step for SGD
            step_key = jax.random.PRNGKey(step)
            shuffled_indices = jax.random.permutation(step_key, len(df_mini))
            df_mini = [df_mini[int(i)] for i in list(shuffled_indices)]

            # ---- train over this Arrow batch ----
            for df in df_mini:
                # skip ahead if resuming
                if last_checkpoint is not False and step <= last_checkpoint:
                    step += 1
                    continue

                stk_moves = df.get_column("moves").to_list()
                stk_moves = jnp.asarray(stk_moves)
                # stk_moves = jnp.squeeze(stk_moves, axis=-1)
                boards = df.get_column("boards").to_list()
                boards = jnp.asarray(boards)

                batch = {"boards": boards, "stk_moves": stk_moves}

                if cfg['type'] == "cls":
                    train_step(model, optimizer, metrics, batch)
                else:
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
                    "Learning_Rate": float(lr_schedule(step-1)),
                })

                print(f"[green]Finished training:[/green] {step:4d}, "
                      f"[cyan]Accuracy:[/cyan] {train_acc:7.4f}, "
                      f"[cyan]Loss:[/cyan] {train_loss:7.4f}")
                metrics.reset()

                # checkpoints
                if step in [64, 256, 1024, 4096, 8192, 12000, 20000, 32768, 64000, 65536] and cfg.get('checkpoint', False):
                    
                    save_checkpoint_modal(model, chk_path / cfg['type'] / cfg['size'] / f"{step}", cfg)

                if step >= cfg['total_steps']:
                    print("[red]Stopping[/red]", step, cfg['total_steps'])
                    break

                step += 1
            
            if step >= cfg['total_steps']:
                print("[red]Stopping[/red]", step, cfg['total_steps'])
                break

    end_time = time.time()
    duration = end_time - start_time
    run.log({"duration": duration})
    run.finish()
    print(f"Training took: {duration:.4f} seconds")

@app.local_entrypoint()
def main():
    # ---------- Config ----------
    with open("config.toml", "rb") as f:
        cfg = tomllib.load(f)
    train_model.remote(cfg, data_file_path, chk_path)
