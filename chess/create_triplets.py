import os
import chess
import chess.pgn
import polars as pl
import pyarrow as pa
import pyarrow.ipc as pa_ipc
from tqdm import tqdm  # pip install tqdm

# ---- Config ----
PARQUET_PATH = "./data/fen_boards/board_move_0.parquet"
PGN_PATH = "./lichess_db_standard_rated_2014-07.pgn"
ARROW_OUT = "./data/fen_boards/board_move_0.arrow"
FEN_COL_IDX = 0   # adjust if needed
MOVE_COL_IDX = 1  # adjust if needed
BUFFER_FLUSH_EVERY = 1024  # how many rows to buffer before writing

# Ensure output dir exists
os.makedirs(os.path.dirname(ARROW_OUT), exist_ok=True)

# ---- Load parquet ----
df = pl.read_parquet(PARQUET_PATH)
n_rows = df.height

def parse_move_from_string(move_str: str, board: chess.Board) -> chess.Move | None:
    # Try UCI (e2e4, e7e8q)
    try:
        mv = chess.Move.from_uci(move_str)
        if mv in board.legal_moves:
            return mv
    except Exception:
        pass

# Counters
parquet_idx = 0
boards_matched = 0
moves_matched = 0
parse_errors = 0

# Arrow schema and writer (streaming)
schema = pa.schema([
    ("fen_board", pa.string()),
    ("stk_moves", pa.string()),
    ("human_moves", pa.string()),
])

# Small buffer to batch writes
buf_fen = []
buf_stk = []
buf_human = []

def flush_buffer(writer):
    if not buf_fen:
        return
    batch = pa.record_batch([
        pa.array(buf_fen, type=pa.string()),
        pa.array(buf_stk, type=pa.string()),
        pa.array(buf_human, type=pa.string()),
    ], schema=schema)
    writer.write(batch)
    buf_fen.clear()
    buf_stk.clear()
    buf_human.clear()

with open(PGN_PATH, "r", encoding="utf-8", errors="ignore") as pgn, \
     pa_ipc.new_file(ARROW_OUT, schema) as arrow_writer, \
     tqdm(total=n_rows, desc="Matching Parquet rows", unit="row") as pbar:

    game_num = 0
    while parquet_idx < n_rows:
        game = chess.pgn.read_game(pgn)
        if game is None:
            break  # no more games

        game_num += 1
        board = game.board()

        for move in game.mainline_moves():
            if parquet_idx >= n_rows:
                break

            fen_parquet, move_parquet_str = df.row(parquet_idx)[FEN_COL_IDX], df.row(parquet_idx)[MOVE_COL_IDX]
            current_fen = board.fen()

            if current_fen == fen_parquet:
                boards_matched += 1

                # Parse parquet move for comparison stats (optional)
                parsed = parse_move_from_string(str(move_parquet_str), board)
                if parsed is None:
                    parse_errors += 1
                else:
                    if parsed == move:
                        moves_matched += 1

                # Capture human-readable SAN for the PGN move at this board
                human_move = chess.Move.uci(move)

                # Buffer the row for Arrow output
                buf_fen.append(fen_parquet)
                buf_stk.append(str(move_parquet_str))
                buf_human.append(human_move)

                # Periodic flush
                if len(buf_fen) >= BUFFER_FLUSH_EVERY:
                    flush_buffer(arrow_writer)

                parquet_idx += 1
                pbar.update(1)
                pbar.set_postfix(games=game_num)

            board.push(move)

    # Final flush for any remaining buffered rows
    flush_buffer(arrow_writer)

# ---- Summary ----
print("----- Summary -----")
print(f"Parquet rows total:           {n_rows}")
print(f"Parquet rows processed:       {parquet_idx}")
print(f"Boards with FEN matched:      {boards_matched}")
print(f"Moves matched (within above): {moves_matched}")
print(f"Parse errors (within above):  {parse_errors}")
print("Status:",
      "All parquet rows processed." if parquet_idx >= n_rows else "Ran out of PGN positions before finishing.")
