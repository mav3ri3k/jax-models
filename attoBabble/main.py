import jax
import jax.numpy as jnp
import optax
import tomllib
import flax.nnx as nnx
from train import train_step
from data import download_shakespeare, load_and_tokenize, batch_iterator,train_tokenizer
from model import Transformer

def main():
    with open("config.toml", "rb") as f:
        cfg = tomllib.load(f)

    if cfg['first'] == True:
        download_shakespeare()
        train_tokenizer()
        
    model = Transformer(cfg=cfg, rngs=nnx.Rngs(cfg['seed']))

    optimizer = nnx.Optimizer(model, optax.adamw(cfg['Lr']))

    all_seqs = load_and_tokenize("input.txt", seq_len=cfg['L'])

    i = 0
    for x_batch, y_batch in batch_iterator(all_seqs, cfg['B'], shuffle=True, seed=cfg['seed']):
        loss = train_step(model,optimizer, x_batch, y_batch) 

        print(i, loss)
        i += 1
main()
