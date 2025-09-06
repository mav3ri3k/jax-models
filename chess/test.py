from checkpoint import save_checkpoint, restore_checkpoint_shard
from rich import print


import orbax.checkpoint as ocp
from pathlib import Path, PosixPath, PurePath
import tomllib

import flax.nnx as nnx
from model import Classifier, EBMChess, EBM
from train import train_step, eval_step, train_step_ebm, eval_step_ebm, model_cls_run 
from tokenizer import Tokenizer
import chess
import jax
import jax.numpy as jnp
import sys


with open("config.toml", "rb") as f:
    cfg = tomllib.load(f)
tokenizer =Tokenizer("")
cfg_small = cfg.copy()
cfg_small['embed_dim'] = 768
cfg_small['ffn_dim'] = 768
cfg_small['layers'] = 12
cfg_small['num_heads'] = 12
model_small = Classifier(cfg_small, rngs=nnx.Rngs(cfg['seed']))
model_xxs = Classifier(cfg, rngs=nnx.Rngs(cfg['seed']))

chk_path = PosixPath("./checkpoints/checkpoint").absolute()
chk_path_small = chk_path / "cls" / "small" / "4096"
chk_path_xxs = chk_path / "cls" / "xxs" / "4096"

def restore(model, chk_path, last_checkpoint):
    try:
        model = restore_checkpoint_shard(model, chk_path, last_checkpoint)
        print(f"[yellow]Model {last_checkpoint} Restored[/yellow]")

        return model
    except UnboundLocalError:
        print(f"[red]last_checkpoint: {last_checkpoint} given in config does not exist[/red]")
        sys.exit()

model_small = restore(model_small, chk_path_small, "4096")
model_xxs = restore(model_xxs, chk_path_xxs, "4096")

fen_board = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
fen_board = tokenizer.encode(fen_board)
board = jnp.asarray(fen_board)[None, :]
logits = model_cls_run(model_small, board)
move = jnp.argmax(logits)
print(tokenizer.decode(move))

logits = model_cls_run(model_xxs, board)
move = jnp.argmax(logits)
print(tokenizer.decode(move))
