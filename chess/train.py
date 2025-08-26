import jax
import jax.numpy as jnp
import sys
import flax.nnx as nnx
import optax
from model import Classifier, EBM, EBMChess

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
# ebm

def loss_fn(model: EBMChess, batch):
  logits = model(batch['boards'])
  loss = optax.softmax_cross_entropy_with_integer_labels(
    logits=logits, labels=batch['stk_moves']
  ).mean()
  return loss, logits

@nnx.jit
def train_step_ebm(model: EBMChess, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch):
  """Train for a single step."""
  grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
  (loss, logits), grads = grad_fn(model, batch)
  metrics.update(loss=loss, logits=logits, labels=batch['stk_moves'])  # In-place updates.
  optimizer.update(grads)  # In-place updates.

@nnx.jit
def eval_step_ebm(model: EBM, metrics: nnx.MultiMetric, batch):
  logits = model(batch['boards'])
  loss = optax.softmax_cross_entropy_with_integer_labels(
    logits=logits, labels=batch['stk_moves']
  ).mean()
  metrics.update(loss=loss, logits=logits, labels=batch['stk_moves'])  # In-place updates.
