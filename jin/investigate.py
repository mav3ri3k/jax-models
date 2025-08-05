import io
import chess
import chess.pgn
import chess.engine
from multiprocessing import Pool, cpu_count

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
PGN_PATH        = "./lichess_db_standard_rated_2014-07.pgn"
STOCKFISH_BIN   = "/usr/local/bin/stockfish"
TIME_PER_MOVE   = 0.01       # seconds per engine query
BATCH_SIZE      = 1         # games per task
MAX_GAMES       = 10_000        # ← set this to e.g. 200 for testing, None for “all 1 000 000”
N_PROCS         = max(1, cpu_count() - 1)
CHUNKSIZE       = 1          # tasks per worker before round‑robin scheduling
# ────────────────────────────────────────────────────────────────────────────────

def init_engine():
    """Launched once per worker process."""
    global engine
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_BIN)

def process_batch(game_texts):
    """Process a list of PGN‑strings, return set of UCI moves."""
    local_uci = set()
    for text in game_texts:
        game = chess.pgn.read_game(io.StringIO(text))
        board = game.board()
        for move in game.mainline_moves():
            result = engine.play(board, chess.engine.Limit(time=TIME_PER_MOVE))
            board.push(move)
            local_uci.add(move.uci())
            local_uci.add(result.move.uci())
    return local_uci

def batch_game_texts(batch_size=BATCH_SIZE, max_games=MAX_GAMES):
    """
    Stream games from disk, yield them in batches of up to `batch_size`.
    Stop after `max_games` games if that is not None.
    """
    with open(PGN_PATH) as f:
        exporter = chess.pgn.StringExporter(
            headers=True, variations=False, comments=False
        )
        batch = []
        count = 0

        while True:
            # stop if we've hit our test limit
            if max_games is not None and count >= max_games:
                if batch:
                    yield batch
                break

            game = chess.pgn.read_game(f)
            if game is None:
                # end of file
                if batch:
                    yield batch
                break

            batch.append(game.accept(exporter))
            count += 1

            if len(batch) >= batch_size:
                yield batch
                batch = []

if __name__ == "__main__":
    all_uci = set()

    with Pool(
        processes=N_PROCS,
        initializer=init_engine
    ) as pool:
        for local_set in pool.imap_unordered(
            process_batch,
            batch_game_texts(batch_size=BATCH_SIZE, max_games=MAX_GAMES),
            chunksize=CHUNKSIZE
        ):
            all_uci.update(local_set)

    print(f"Processed up to {MAX_GAMES or 'all'} games; distinct UCI moves = {len(all_uci)}")
