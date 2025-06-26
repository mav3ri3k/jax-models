import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
from model import VisionTransformer

def loss_fn(model: VisionTransformer, batch):
  pred_eps = model(batch['eps'], batch['times'], batch['labels'])

  loss = optax.l2_loss(
    predictions=pred_eps, targets=batch['eps']
  ).mean()
  return loss, pred_eps

@nnx.jit
def train_step(model: VisionTransformer, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch):
  """Train for a single step."""  
  grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
  (loss, pred_eps), grads = grad_fn(model, batch)
  metrics.update(loss=loss)  # In-place updates.
  optimizer.update(grads)  # In-place updates.

@nnx.jit
def eval_step(model: VisionTransformer, metrics: nnx.MultiMetric, batch):
  loss, logits = loss_fn(model, batch)
  metrics.update(loss=loss, logits=logits, labels=batch['labels'])  # In-place updates.
