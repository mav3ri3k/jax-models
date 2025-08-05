import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
from model import Classifier, EBM

def loss_fn(model: Classifier, batch):
  logits = model(batch['boards'])
  loss = optax.softmax_cross_entropy_with_integer_labels(
    logits=logits, labels=batch['moves']
  ).mean()
  return loss, logits

@nnx.jit
def train_step(model: Classifier, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch):
  """Train for a single step."""
  grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
  (loss, logits), grads = grad_fn(model, batch)
  metrics.update(loss=loss, logits=logits, labels=batch['moves'])  # In-place updates.
  optimizer.update(grads)  # In-place updates.

@nnx.jit
def eval_step(model: Classifier, metrics: nnx.MultiMetric, batch):
  loss, logits = loss_fn(model, batch)
  metrics.update(loss=loss, logits=logits, labels=batch['moves'])  # In-place updates.

#---  
def loss_fn_ebm(model: EBM, batch):
  logits = model(batch['boards'])
  # print(f"{logits.shape} {batch['moves'].shape}")
  loss = optax.softmax_cross_entropy_with_integer_labels(
    logits=logits, labels=batch['moves']
  ).mean()
  return loss, logits

@nnx.jit
def train_step_ebm(model: EBM, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch):
  """Train for a single step."""
  grad_fn = nnx.value_and_grad(loss_fn_ebm, has_aux=True)
  (loss, logits), grads = grad_fn(model, batch)
  metrics.update(loss=loss, logits=logits, labels=batch['moves'])  # In-place updates.
  optimizer.update(grads)  # In-place updates.

@nnx.jit
def eval_step_ebm(model: EBM, metrics: nnx.MultiMetric, batch):
  loss, logits = loss_fn_ebm(model, batch)
  metrics.update(loss=loss, logits=logits, labels=batch['moves'])  # In-place updates.
