from collections import Counter
import chess.pgn
from tqdm import tqdm
import math
import re

def bucket_label(move_count, bucket_size):
    """
    Given a move count (full moves) and bucket size,
    returns a string label like "1-10", "11-20", etc.
    If move_count <= 0, returns "0".
    """
    if move_count <= 0:
        return "0"
    idx   = (move_count - 1) // bucket_size
    lower = idx * bucket_size + 1
    upper = lower + bucket_size - 1
    return f"{lower}-{upper}"

def parse_pgn_stats(pgn_path, max_games=None, bucket_size=10):
    """
    Parse up to max_games from pgn_path, count full moves,
    and tally into buckets of size bucket_size.
    """
    buckets      = Counter()
    games_parsed = 0
    pbar         = tqdm(total=max_games, desc="Parsing games", unit="game")

    with open(pgn_path, encoding="utf-8") as pgn_file:
        while True:
            if max_games is not None and games_parsed >= max_games:
                break

            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break  # EOF

            # count half‐moves and convert to full moves
            half_moves = sum(1 for _ in game.mainline_moves())
            full_moves = math.ceil(half_moves / 2)

            label = bucket_label(full_moves, bucket_size)
            buckets[label] += 1

            games_parsed += 1
            pbar.update(1)

    pbar.close()
    return buckets

if __name__ == "__main__":
    PGN_FILE   = "./lichess_db_standard_rated_2014-07.pgn"
    MAX_GAMES  = None      # or None to parse all games
    BUCKET_SIZE = 10       # e.g. 1–10, 11–20, …

    stats = parse_pgn_stats(PGN_FILE, max_games=MAX_GAMES, bucket_size=BUCKET_SIZE)

    print(f"\nParsed up to {MAX_GAMES or 'all'} games. Move-count buckets:")
    # sort by the first integer in the bucket label
    def sort_key(item):
        label = item[0]
        # grab the first group of digits
        m = re.match(r"(\d+)", label)
        return int(m.group(1)) if m else 0

    for bucket, count in sorted(stats.items(), key=sort_key):
        print(f"  {bucket:>7}: {count}")
