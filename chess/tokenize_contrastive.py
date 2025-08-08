from tokenizer import Tokenizer
import polars as pl
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os
import pyarrow as pa
import pyarrow.ipc as ipc

# -------- Config --------
input_path  = "./data/fen_boards/board_move_0.arrow"          # <- read from the Arrow IPC created earlier
output_path = "./data/pre_tokenized/cache_tokenized_triplet.arrow"
tokenizer   = Tokenizer("None")
batch_size  = 1024
num_workers = 4  # tune to CPU cores

os.makedirs(os.path.dirname(output_path), exist_ok=True)

# -------- Batch processor --------
def process_batch(df: pl.DataFrame) -> pl.DataFrame:
    tok_fens   = []
    tok_stk    = []
    tok_human  = []
    # Expect columns: fen_board, stk_moves, human_moves
    for row in df.iter_rows(named=True):
        try:
            fen_ids   = tokenizer.encode(row["fen_board"])
            stk_ids   = tokenizer.encode(row["stk_moves"])
            human_ids = tokenizer.encode(row["human_moves"])
            tok_fens.append(fen_ids)
            tok_stk.append(stk_ids)
            tok_human.append(human_ids)
        except AssertionError as e:
            print(e)
            # skip bad rows consistently with your previous code
            # continue

    return pl.DataFrame(
        {
            "fen_board":  tok_fens,
            "stk_moves":  tok_stk,
            "human_moves": tok_human,
        }
    )

# -------- Streaming read (IPC/.arrow) + parallel tokenize + streaming write (IPC) --------
lazy_df = pl.scan_ipc(input_path)   # lazy scan from Arrow IPC
with open(output_path, "wb") as f_out:
    writer = None
    offset = 0
    futures = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # enqueue chunks
        while True:
            df_chunk = lazy_df.slice(offset, batch_size).fetch()
            if df_chunk.is_empty():
                break
            futures.append(executor.submit(process_batch, df_chunk))
            offset += batch_size

        # collect + write as they finish
        with tqdm(total=len(futures), desc="Tokenizing Batches", unit="batch") as pbar:
            for fut in as_completed(futures):
                df_out = fut.result()
                if df_out.is_empty():
                    pbar.update(1)
                    continue

                table = df_out.to_arrow()  # columns preserved: fen_board, stk_moves, human_moves

                if writer is None:
                    writer = ipc.RecordBatchStreamWriter(f_out, table.schema)

                writer.write_table(table)
                pbar.update(1)

    if writer:
        writer.close()
