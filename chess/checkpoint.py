import orbax.checkpoint as ocp
import tomllib
import flax.nnx as nnx
import os
import jax
import jax.numpy as jnp
from rich import print
from rich.prompt import Prompt
from upload import upload, download_folder

try:
    import google.colab
    colab = True
except ImportError:
    colab = False

def save_checkpoint(model, rel_path, cfg):
    # rel_path = "./checkpoints/one"
    ckpt_dir = os.path.abspath(rel_path)

    _, state = nnx.split(model)

    with ocp.StandardCheckpointer() as checkpointer:
        try:
            checkpointer.save(ckpt_dir, state)
            print("[green]Checkpoint saved[/green]")
        except ValueError:
            choice = Prompt.ask(f"Checkpoint: {rel_path} already exist. Force ?", choices=["y", "n"], default="n")
            if choice == 'y':
                ckpt_dir = ocp.test_utils.erase_and_create_empty(ckpt_dir)
                checkpointer.save(ckpt_dir, state)
                print("[green]Checkpoint saved[/green]")
            else:
                print("[red]Checkpoint save aborted[/red]")
                return False
    if cfg['upload']:
        upload(rel_path)
    return True

def restore_checkpoint(model, rel_path, step):
    ckpt_dir = os.path.abspath(rel_path)
    ckpt_dir = ocp.test_utils.erase_and_create_empty(ckpt_dir)

    download_folder(f"{step}.json", rel_path)

    abstract_model = nnx.eval_shape(lambda: model)
    print(abstract_model)
    graphdef, abstract_state = nnx.split(abstract_model)
    print(abstract_state)

    checkpointer = ocp.StandardCheckpointer()
    try:
        state_restored = checkpointer.restore(
            ckpt_dir,
            abstract_state
        )
        print(state_restored)
    except FileNotFoundError as e:
        print(e)

    return nnx.merge(graphdef, state_restored)
