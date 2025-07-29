import polars as pl
import pyarrow as pa

import pyarrow.ipc as ipc
import pandas as pd
from tqdm import tqdm
import jax.numpy as jnp

df = pl.read_parquet("./data/tokenizer_output.parquet")
print(df)
print(df.schema, df.height)
