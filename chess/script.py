import polars as pl

# Path to your Arrow file
file_path = "./data/pre_tokenized/cache_tokenized_triplet.arrow"

# Read the Arrow file
df = pl.read_ipc(file_path)  # For Arrow IPC/Feather files

# Print the first 10 rows
for row in df.head(10).iter_rows(named=True):
    print(row)
