import orbax.checkpoint as ocp
import tomllib
import flax.nnx as nnx
import os
import jax
import jax.numpy as jnp
from rich import print
from rich.prompt import Prompt
from upload import upload, download_folder
import jax.sharding as jshard

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

def save_checkpoint_modal(model, path, cfg):
    # rel_path = "./checkpoints/one"
    ckpt_dir = path
    _, state = nnx.split(model)

    with ocp.StandardCheckpointer() as checkpointer:
        try:
            checkpointer.save(ckpt_dir, state)
            print("[green]Checkpoint saved[/green]")
        except ValueError:
            if cfg['chk_force_save']:
                ckpt_dir = ocp.test_utils.erase_and_create_empty(ckpt_dir)
                if os.path.isdir(ckpt_dir):
                    os.rmdir(ckpt_dir)
                checkpointer.save(ckpt_dir, state)
                print("[green]Checkpoint saved[/green]")
            else:
                print("[red]Checkpoint save aborted[/red]")
                return False
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


def restore_checkpoint_shard(model, path, step):
    from jax.sharding import Mesh
    import numpy as np
    mesh = Mesh(devices=np.array(jax.devices()),
            axis_names=('model'))
    ckpt_dir = path

    abstract_model = nnx.eval_shape(lambda: model)
    graphdef, abstract_state = nnx.split(abstract_model)
    abstract_state = nnx.state(abstract_model)
    print(type(abstract_state))
    # print(abstract_state)

    device = jax.local_devices()[0]
    sharding = jshard.SingleDeviceSharding(device)

    args = ocp.handlers.StandardRestoreArgs(strict=False, support_layout=True)
    checkpointer = ocp.StandardCheckpointer()

    try:
        # spec = nnx.get_partition_spec(abstract_state)
        # print("test")
        # sharding = jax.tree.map(lambda p: jax.sharding.NamedSharding(mesh, p), spec)
        # print("test")
        sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
        spec = nnx.get_partition_spec(abstract_state)
        sharding = jax.tree.map(lambda p: jax.sharding.NamedSharding(mesh, p), spec)
        nnx_shard =  sharding
        # print(nnx_shard)
        # nnx_shard = nnx.get_named_sharding(absract_state, mesh)
        # print("test")
        # sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
        abstract_st = jax.tree.map(
          lambda a, s: jax.ShapeDtypeStruct(a.shape, a.dtype, sharding=s),
          abstract_state, 
          nnx_shard
        )

        state_restored = checkpointer.restore(
            ckpt_dir,
            abstract_st,
            strict=True
        )
        # print(state_restored)
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(e)

    return nnx.merge(graphdef, state_restored)

def restore_checkpoint_modal(model, path, step):
    ckpt_dir = path

    abstract_model = nnx.eval_shape(lambda: model)
    graphdef, abstract_state = nnx.split(abstract_model)

    checkpointer = ocp.StandardCheckpointer()
    try:
        state_restored = checkpointer.restore(
            ckpt_dir,
            abstract_state
        )
        # print(state_restored)
    except FileNotFoundError as e:
        print(e)

    return nnx.merge(graphdef, state_restored)
