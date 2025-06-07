import os
import sys
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import time
import optax
import tomllib

import orbax.checkpoint as ocp
from orbax.checkpoint import CheckpointManager, CheckpointManagerOptions, args

from data import load_data
from model import VisionTransformer
from train import train_step, eval_step

def main():
    data = load_data()
    train_images = jnp.asarray(data["train_images"])  # shape (50000,32,32,3), dtype uint8
    train_labels = jnp.asarray(data["train_labels"])  # shape (50000,), dtype int64
    test_images  = jnp.asarray(data["test_images"])   # shape (10000,32,32,3)
    test_labels  = jnp.asarray(data["test_labels"])   # shape (10000,)

    with open("config.toml", "rb") as f:
        cfg = tomllib.load(f)

    model = VisionTransformer(cfg, rngs=nnx.Rngs(cfg['seed']))
    optimizer = nnx.Optimizer(model, optax.adamw(cfg['learning_rate']))

    metrics = nnx.MultiMetric(
      accuracy=nnx.metrics.Accuracy(),
      loss=nnx.metrics.Average('loss'),
    )

    # 3) Set up checkpoint directory & manager
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    options = CheckpointManagerOptions(max_to_keep=3)
    manager = CheckpointManager(
        checkpoint_dir,
        options=options,
        item_names=('model', 'optimizer'),
    )

    # 4) Try to restore latest
    latest = manager.latest_step()
    if latest is not None:
        restored = manager.restore(
            latest,
            # For restore, no args needed when using item_names + StandardSave
        )
        model, optimizer = restored.model, restored.optimizer
        print(f"🔄 Restored from epoch {latest}")
    else:
        print("✨ No checkpoint found, starting fresh")

    # 5) If eval-only, run test and exit
    if cfg.get('eval', False):
        # abort if checkpoint not available
        if latest is None:
            print("✨ No checkpoint found, eval aborted")
            sys.exit(0)

        print("🛠  Eval-only mode")
        test_metrics = nnx.MultiMetric(
            accuracy=nnx.metrics.Accuracy(),
            loss=nnx.metrics.Average('loss'),
        )
        bs = cfg['batch_size']
        for i in range(0, test_images.shape[0], bs):
            batch = {
                "images": test_images[i : i+bs],
                "labels": test_labels[i : i+bs],
            }
            eval_step(model, test_metrics, batch)
        print("📊 Test →", test_metrics.compute())
        sys.exit(0)

    for epoch in range(cfg["epochs"]):
        num_train = train_images.shape[0]
        batch_size = cfg['batch_size']

        perm = jax.random.permutation(jax.random.PRNGKey(epoch), num_train)
        shuffled_imgs = train_images[perm]
        shuffled_lbls = train_labels[perm]

        batches = [
            {"images": shuffled_imgs[i : i + batch_size],
             "labels": shuffled_lbls[i : i + batch_size]}
            for i in range(0, num_train, batch_size)
        ]

        # time a epoch
        start_time = time.time()
        for batch in batches:
            train_step(model, optimizer, metrics, batch)
        print(f"Epoch {epoch+1} →", metrics.compute())
        end_time = time.time()
        duration = end_time - start_time
        print(f"Model training took: {duration:.4f} seconds")

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

        print(f" Test  {epoch+1} →", test_metrics.compute())

        # # — checkpoint save —
        # manager.save(
        #     epoch,
        #     args=ocp.args.Composite(
        #         model=ocp.args.StandardSave(model),
        #         optimizer=ocp.args.StandardSave(optimizer),
        #     ),
        # )
        # manager.wait_until_finished()
        # print(f"💾 Checkpointed epoch {epoch}")

