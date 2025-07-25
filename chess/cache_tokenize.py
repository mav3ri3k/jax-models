from tokenizer import Tokenizer
import polars as pl
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os
import pyarrow as pa
import pyarrow.ipc as ipc

input_path = "./data/fen_boards/board_move_0.parquet"
output_path = "./data/pre_tokenized/cache_tokenized.arrow"
tokenizer = Tokenizer("None")
batch_size = 1_000
num_workers = 4  # Adjust based on CPU cores

# Mock tokenizer (replace with your real one)
def process_batch(df: pl.DataFrame) -> pl.DataFrame:
    tokenized_boards = []
    tokenized_moves = []
    for row in df.iter_rows(named=True):
        try:
            a = tokenizer.encode(row["fen_boards"])
            b = tokenizer.encode(row["stk_moves"])
            tokenized_boards.append(a)
            tokenized_moves.append(b)
        except AssertionError:
            continue
    return pl.DataFrame({
        "boards": tokenized_boards,
        "moves": tokenized_moves
    })

# Initialize lazy read
lazy_df = pl.scan_parquet(input_path)

# Open output Arrow file for streaming
with open(output_path, "wb") as f_out:
    writer = None
    offset = 0

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []

        # Submit batch read/processing jobs
        while True:
            df_chunk = lazy_df.slice(offset, batch_size).fetch()
            if df_chunk.is_empty():
                break

            futures.append(executor.submit(process_batch, df_chunk))
            offset += batch_size

        # Collect results in order of completion
        with tqdm(total=len(futures), desc="Processing Batches") as pbar:
            for future in as_completed(futures):
                df_out = future.result()
                if writer is None:
                    writer = ipc.RecordBatchStreamWriter(f_out, df_out.to_arrow().schema)

                # Write this batch
                writer.write_table(df_out.to_arrow())

                pbar.update(1)
    
    if writer:
        writer.close()

