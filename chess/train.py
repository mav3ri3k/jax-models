import jax
import jax.numpy as jnp
import sys
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
  energy_pos = jnp.take_along_axis(logits, batch['stk_moves'], axis=-1)
  energy_neg = jnp.take_along_axis(logits, batch['human_moves'], axis=-1)

  energy_all = jnp.concatenate([energy_pos, energy_neg], axis=-1)
  labels = jnp.zeros(energy_all.shape[0], dtype=jnp.int32)
  
  # print(f"{logits.shape} {batch['moves'].shape}")
  loss = optax.softmax_cross_entropy_with_integer_labels(
    logits=energy_all, labels=labels
  ).mean()
  return loss, energy_all

@nnx.jit
def train_step_ebm(model: EBM, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch):
  """Train for a single step."""
  grad_fn = nnx.value_and_grad(loss_fn_ebm, has_aux=True)
  (loss, logits), grads = grad_fn(model, batch)
  labels = jnp.zeros(logits.shape[0], dtype=jnp.int32)
  metrics.update(loss=loss, logits=logits, labels=labels)  # In-place updates.
  optimizer.update(grads)  # In-place updates.

@nnx.jit
def eval_step_ebm(model: EBM, metrics: nnx.MultiMetric, batch):
  logits = model(batch['boards'])
  labels = jnp.squeeze(batch['stk_moves'], axis=-1)
  loss = optax.softmax_cross_entropy_with_integer_labels(
    logits=logits, labels=labels
  ).mean()
  metrics.update(loss=loss, logits=logits, labels=labels)  # In-place updates.
