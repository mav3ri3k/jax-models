This is a small language model meant for experimenting.

# Get started
```bash
git clone https://github.com/mav3ri3k/jax-models.git
cd ./jax-models/attoBabble/ && uv sync
uv run main.py
```

# Use GPU/TPU
By default, the project will use CPU, to use GPU/TPU:

In `pyproject.toml`

- On `GPU`, change `jax` -> `jax[cuda]`
  ```bash
  -  "jax==0.4.38",
  +  "jax[cuda]",
  ```
- On `TPU`, change `jax` -> `jax[tpu]`
  ```bash
  -  "jax==0.4.38",
  +  "jax[tpu]",
  ```
Run `uv sync` again

# Prompts to Try
ROMEO:
HAMLET:

What news, good friend?
She wept, yet smiled again, for
A knocking at the gate.

Love is a
Fortune smiles upon

Email from the king:
The robot said,
