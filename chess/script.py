import polars as pl
from tqdm import tqdm
from tokenizer import Tokenizer

# Path to your Arrow file
file_path = "./records.parquet"

file_path_tk = "./data/tokenizer.parquet"

tk = Tokenizer("")
scan = pl.scan_parquet("records.parquet").select(["board", "move"])
df = pl.read_parquet(file_path_tk)
for row in df.iter_rows(named=True):
    print(row)
print(df)
print(df.schema)
from typing import List, Set
import chess
import logging as lg

tk = Tokenizer("")
scan = pl.scan_parquet("records.parquet").select(["board", "move"])
row_count = pl.scan_parquet(file_path).select(pl.len()).collect().item()
print("Row count:", row_count)
i = 0
for offset in tqdm(range(0, row_count, 512)):  # example slicing into 512 batches
    # if i > 100:
    #     break
    # i += 1
    batch = scan.slice(offset, 512).collect(streaming=True)
    for row in batch.iter_rows(named=True):
        board = row["board"]
        move = row["move"]

        encoded_board = tk.encode(board)  # or pass as tuple/list depending on your tokenizer
        try:
            encoded_move = tk.encode(move)
        except Exception as e:
            print(e)
            
