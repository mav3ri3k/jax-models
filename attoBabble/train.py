from model import Transformer
import optax
import flax.nnx as nnx
import jax.numpy as jnp
import jax

def loss_fn(model: Transformer, x_batch: jnp.ndarray, y_batch: jnp.ndarray):
    logits = model(x_batch)
    cross_entropy = optax.losses.softmax_cross_entropy_with_integer_labels(logits=logits, labels=y_batch)

    return jnp.mean(cross_entropy)

@nnx.jit
def train_step(model: Transformer, optimizer: nnx.Optimizer, x_batch, y_batch):
    grad_fn = nnx.value_and_grad(loss_fn)
    loss, grads = grad_fn(model, x_batch, y_batch)
    optimizer.update(grads)

    return loss

