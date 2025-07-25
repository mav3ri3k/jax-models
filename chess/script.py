import polars as pl
import pyarrow as pa

import pyarrow.ipc as ipc
import pandas as pd

with open("./data/pre_tokenized/cache_tokenized.arrow", "rb") as f:
    reader = ipc.RecordBatchStreamReader(f)
    # Read all batches and convert to a PyArrow Table
    table = reader.read_all()
    # Convert to pandas
    df = pl.from_arrow(table)
    # df = table.to_pandas()
    print(df.head(5))
    print(type(df))
    print(df.height)
