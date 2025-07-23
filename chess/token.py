import chess
import time
import chess.pgn
import os
import logging
import polars as pl
from typing import List, Iterable
from joblib import Parallel, delayed


lg = logging.getLogger(__name__)
lg.setLevel(logging.DEBUG)
# logging.disable(logging.CRITICAL)
# logging.disable(logging.NOTSET)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    '%(lineno)d - %(name)s - %(levelname)s - %(message)s'
)
handler.setFormatter(formatter)
if not lg.handlers:
    lg.addHandler(handler)


class Tokenizer:
    @classmethod
    def _builder_fen(self, pgn_source, n_jobs: int = -1) -> List:

        batch_size = 1
        limit = 1_00
        lg.info("Fen Builder Start")

        # ---------- 1) Fast scan: get byte offsets for each game ----------
        offsets: list[int] = []
        with open(pgn_source, "r", encoding="utf-8", errors="replace") as f:
            while True:
                off = f.tell()
                headers = chess.pgn.read_headers(f)  # skips body, moves file pointer to next game
                if headers is None:
                    break
                offsets.append(off)
                if limit is not None and len(offsets) >= limit:
                    break

        lg.info(f"Found {len(offsets)} game offsets; parallelizing on {n_jobs} cores")

        # ---------- 2) Helpers ----------
        def _chars_from_offset_batch(offset_batch: Iterable[int]) -> set[str]:
            """Worker: open file, seek to each offset, parse, extract FEN chars."""
            local_chars: set[str] = set()
            # Re-open in each process to avoid pickling file handles.
            with open(pgn_source, "r", encoding="utf-8", errors="replace") as f:
                for off in offset_batch:
                    f.seek(off)
                    game = chess.pgn.read_game(f)
                    if game is None:
                        continue
                    board = game.board()
                    for mv in game.mainline_moves():
                        board.push(mv)
                        # You were building a *character* set from FEN strings:
                        local_chars.update(board.fen())
            return local_chars

        # Slice offsets into batches so each task amortizes file open/seek cost
        def _chunk(seq, size):
            for i in range(0, len(seq), size):
                yield seq[i:i+size]

        # ---------- 3) Run in parallel and union ----------
        all_chars: set[str] = set()
        for chunk_set in Parallel(n_jobs=n_jobs)(
            delayed(_chars_from_offset_batch)(batch)
            for batch in _chunk(offsets, batch_size)
        ):
            all_chars.update(chunk_set)

        lg.info("Fen Builder Finish")

        return sorted(all_chars)
        
    @classmethod
    def _builder_move(self) -> List:
        lg.info("Move Builder Start")
        all_uci = set()
        # Iterate over every possible source square
        for square in chess.SQUARES:
            piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP,
                           chess.ROOK, chess.QUEEN, chess.KING]
            for pt in piece_types:
                board = chess.Board.empty()
                board.set_piece_at(square, chess.Piece(pt, chess.WHITE))
                # every legal move for that single piece
                for move in board.legal_moves:
                    all_uci.add(move.uci())
        # promotions
        # promotions – captures and non-captures, BOTH colours
        for r_from, r_to in [(1, 0), (6, 7)]:          # White & Black
            for f in range(8):
                # straight
                all_uci.add(chess.Move(
                    chess.square(f, r_from),
                    chess.square(f, r_to),
                    chess.QUEEN
                ).uci())
                # captures left & right
                for df in (-1, 1):
                    f_to = f + df
                    if 0 <= f_to < 8:
                        for promo in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
                            all_uci.add(chess.Move(
                                chess.square(f, r_from),
                                chess.square(f_to, r_to),
                                promo
                            ).uci())
        # 24 castling strings that exist in the paper’s vocabulary
        # sort
        lg.info("Move Builder Finish")
        return sorted(all_uci)

    def __init__(self, pgn_source, force_new = False):
        lg.info("Tokenizer Initialized")
        if not os.path.isfile("./data/tokenizer.avro") or force_new:
            all_uci = self._builder_move()
            start_time = time.time()
            all_chars = self._builder_fen(pgn_source)
            duration = time.time() - start_time

            lg.info(f"Fen Builder took {duration}")

            idx = []
            tokens = []
            for id, token in enumerate(all_chars + all_uci):
                idx.append(id)
                tokens.append(token)
            vocab = {"idx": idx, "tokens": tokens}

            df = pl.DataFrame(vocab)
            df.write_avro("./data/tokenizer.avro")

    def encode(self, token: str) -> int:
        df = pl.read_avro("./data/tokenizer.avro")
        df_filter = df.filter(pl.col("tokens") == token)
        assert df_filter.height == 1

        return df_filter["idx"].item()

    def decode(self, token_id: int) -> str:
        df = pl.read_avro("./data/tokenizer.avro")
        df_filter = df.filter(pl.col("idx") == token_id)
        assert df_filter.height == 1

        return df_filter["tokens"].item()

if __name__ == "__main__":
    tokenizer = Tokenizer(pgn_source="./lichess_db_standard_rated_2014-07.pgn", force_new = True)

    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    for c in fen:
        token_id = tokenizer.encode(c)
        token = tokenizer.decode(token_id)
        print(f"Id: {token_id}, token: {token}")
    print()
    uci = "e2e4"
    token_id = tokenizer.encode(uci)
    token = tokenizer.decode(token_id)
    print(f"Id: {token_id}, token: {token}")
