import polars as pl
import pyarrow as pa

import pyarrow.ipc as ipc
import pandas as pd
from tqdm import tqdm
import jax.numpy as jnp
from checkpoint import save_checkpoint
from upload import upload

upload("./checkpoints/32")


