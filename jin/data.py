import logging
import time
from typing import Iterable, List, Tuple, Set
from joblib import Parallel, delayed
from tqdm import tqdm
import polars as pl
import chess
import chess.pgn
import xxhash
from pathlib import Path
from tokenizer import Tokenizer

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
STOCKFISH_PATH   = "/usr/local/bin/stockfish"
PGN_PATH   = "./lichess_db_standard_rated_2014-07.pgn"
LIMIT_GAMES = None          # None ⇒ all
BATCH_SIZE = 100_000
CHUNK_GAMES = 64              # games per worker task
N_PROCS     = -1              # -1 ⇒ all cores
# ────────────────────────────────────────────────────────────────────────────────

lg = logging.getLogger(__name__)
lg.setLevel(logging.DEBUG)
if not lg.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(lineno)d - %(name)s - %(levelname)s - %(message)s'))
    lg.addHandler(handler)

def _chunk(seq: List[int], size: int):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]

# ─── Core ───────────────────────────────────────────────────────────────────────
def _scan_game_offsets(pgn_path: str, limit: int | None) -> List[int]:
    """Fast pass: collect byte offsets of each PGN game."""
    offsets: List[int] = []
    with open(pgn_path, "r", encoding="utf-8", errors="replace") as f:
        while True:
            off = f.tell()
            headers = chess.pgn.read_headers(f)  # advances file handle
            if headers is None:
                break
            offsets.append(off)
            if limit is not None and len(offsets) >= limit:
                break
    return offsets

def _worker_extract_unique_fens(pgn_path: str, offset_batch: Iterable[int]) -> Tuple[int, List[str], Set[bytes]]:
    """
    Worker:
      - Parse each game starting at each offset
      - Walk mainline moves, build fixed-length FEN strings
      - Deduplicate locally with a hash set
    Returns:
      (positions_seen, unique_fens_list, unique_hashes_set)
    """
    local_fens: List[str] = []
    local_hashes: Set[bytes] = set()
    positions_seen = 0

    with open(pgn_path, "r", encoding="utf-8", errors="replace") as f:
        for off in offset_batch:
            f.seek(off)
            game = chess.pgn.read_game(f)
            if game is None:
                continue

            board = game.board()
            for mv in game.mainline_moves():
                positions_seen += 1

                fen_fixed = board.fen()
                h = xxhash.xxh64(fen_fixed).digest()
                if h not in local_hashes:
                    local_hashes.add(h)
                    local_fens.append(fen_fixed)

                board.push(mv)  # move to next position

    return positions_seen, local_fens, local_hashes

def _unique_fen_builder(pgn_source: str = PGN_PATH,
                       limit: int | None = LIMIT_GAMES,
                       n_jobs: int = N_PROCS,
                       chunk_games: int = CHUNK_GAMES) -> List[str]:
    lg.info("Unique FEN builder start")

    offsets = _scan_game_offsets(pgn_source, limit)
    lg.info(f"Found {len(offsets)} game offsets; parallelizing on {n_jobs} cores")

    batch_size = BATCH_SIZE
    batch_offsets = []

    i_batch = 0
    while True:
        if len(offsets) - len(batch_offsets)*batch_size < batch_size:
            batch_offsets.append(offsets[batch_size*i_batch:])
            break
        
        batch_offsets.append(offsets[batch_size*i_batch : batch_size*(i_batch + 1)])
        i_batch += 1
        
    no_batches = len(batch_offsets)
    lg.info(f"{no_batches} batches created")


    total_seen = 0
    for idx, offsets in enumerate(batch_offsets):
        results = Parallel(n_jobs=n_jobs, batch_size="auto", prefer="processes")(
            delayed(_worker_extract_unique_fens)(pgn_source, batch)
            for batch in _chunk(offsets, chunk_games)
        )

        all_fens: List[str] = []
        all_hashes: Set[bytes] = set()

        for seen, fens, hashes in results:
            total_seen += seen
            # Merge uniques
            for fen, h in zip(fens, hashes):
                if h not in all_hashes:
                    all_hashes.add(h)
                    all_fens.append(fen)

        assert len(all_fens) == len(all_hashes), (
            f"Unique counts mismatch: fens={len(all_fens)}, hashes={len(all_hashes)}"
        )

        results = {"unique_fen_boards": all_fens}
        df = pl.DataFrame(results)
        df.write_parquet(f"./data/unique_boards/unique_boards_{idx}.parquet")
        lg.info(f"Saved board {idx}")

    lg.info(f"Positions seen: {total_seen}")
    lg.info("Unique FEN builder complete")

    return all_fens

def eval_chunk(fens, movetime_s=0.05):
    """Evaluate a list of FENs with a single Stockfish instance."""
    engine = chess.engine.SimpleEngine.popen_uci(str(STOCKFISH_PATH))
    # Keep each engine single-threaded so parallelism scales

    out = []
    limit = chess.engine.Limit(time=movetime_s)
    for fen in fens:
        board = chess.Board(fen)
        result = engine.play(board, limit)
        out.append((fen, result.move.uci()))
    engine.quit()
    return out

# ---- main ------------------------------------------------------------------

def stockfish_best_moves(count, movetime_ms: int = 50):
    lg.info("Start stockfish")
    start_time = time.time()

    total_done = 0  # across all files

    for file_idx in range(count):
        df = pl.read_parquet(f"./data/unique_boards/unique_boards_{file_idx}.parquet")
        fens = df["unique_fen_boards"].to_list()

        mini_batch_limit = None
        chunk_limit = 10_000
        # build chunks once so we know how many there are
        chunks = [fens[i:i+CHUNK_GAMES] for i in range(0, len(fens), CHUNK_GAMES)]
        if mini_batch_limit != None:
            chunks = [chunk[:mini_batch_limit] for chunk in chunks]
        n_mini_batch = len(chunks[0])
        lg.info(f"One mini batch has {n_mini_batch} boards")

        if chunk_limit != None:
            chunks = chunks[:chunk_limit]
        n_chunks = len(chunks)
        lg.info(f"File {file_idx}: {n_chunks} mini-batches")

        all_fens, all_moves = [], []

        for chunk_results in Parallel(n_jobs=N_PROCS)(
                delayed(eval_chunk)(chunk, movetime_ms / 1000.0)
                for chunk in tqdm(chunks)
        ):
            total_done += 1

            for fen, move in chunk_results:
                all_fens.append(fen)
                all_moves.append(move)

        lg.info("Stockfish move estimates completed for file %d", file_idx)
        pl.DataFrame({"fen_boards": all_fens, "stk_moves": all_moves}) \
          .write_parquet(f"./data/fen_boards/board_move_{file_idx}.parquet")

    lg.info("End stockfish, took: %.2fs", time.time() - start_time)


def prepare_data(force_boards=False, force_moves=False, force_tokenization=False):
    folder = Path("./data/unique_boards")
    if not any(folder.iterdir()) or force_boards:
        _unique_fen_builder(PGN_PATH)
        lg.info("Parquet file written for unique boards")
    else:
        lg.info("Using pre-computed parquet files for unique boards")

    folder = Path("./data/fen_boards")
    if not any(folder.iterdir()) or force_moves:
        stockfish_best_moves(1, movetime_ms=50)
        lg.info("Parquet file written for board-move pair")
    else:
        lg.info("Using pre-computed parquet files for board-move pair")


# prepare_data(force_boards=False, force_moves=True, force_tokenization=True)

def load_data(pgn_source, force_new = False):
    f =  open("./data/pre_tokenized/cache_tokenized.arrow", "rb")
    reader = ipc.RecordBatchStreamReader(f)
    # Read all batches and convert to a PyArrow Table
    i = 0
    total = 0
    flag = True
    try:
        while True:
            i += 1
            batch = reader.read_next_batch()
            df = pl.from_arrow(batch)
            total += df.height

            if flag:
                flag = not flag
                print(df.shape)
                print(df.height)
    except StopIteration:
        pass
    print(i)
    print(total)
    f.close()
