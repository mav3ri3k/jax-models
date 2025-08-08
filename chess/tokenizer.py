import chess
import time
import chess.pgn
import os
import logging
from tqdm import tqdm  # pip install tqdm
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
    @staticmethod
    def _pad_right(s: str, length: int) -> str:
        return (s + '.' * length)[:length]

    @classmethod
    def _pad_fen(self, fen: str) -> str:
        board, turn, castling, ep, halfmove, fullmove = fen.split()

        # 1–64 : piece placement (8*8 = 64)
        board = board.replace('/', '')               # remove '/'
        board = self._pad_right(board, 64)

        # 66–69  : castling (KQkq), padded right with '.'
        castling = self._pad_right(castling, 4)

        # 70–71  : en-passant square ('-' or two chars like 'e3')
        ep = self._pad_right(ep, 2)

        # 72–73  : halfmove clock (2 digits, zero-padded)
        halfmove = self._pad_right(halfmove, 2)

        # 74–76  : fullmove number (3 digits, zero-padded)
        fullmove = self._pad_right(fullmove, 3)

        fixed = board + turn + castling + ep + halfmove + fullmove
        assert len(fixed) == 76, f"Length {len(fixed)} != {76}"
        return fixed

    @classmethod
    def _builder_fen(self, pgn_source, n_jobs: int = -1) -> List:

        batch_size = 1
        limit = None
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
                        local_chars.update(self._pad_fen(board.fen()))
            return local_chars

        # Slice offsets into batches so each task amortizes file open/seek cost
        def _chunk(seq, size):
            for i in range(0, len(seq), size):
                yield seq[i:i+size]

        # ---------- 3) Run in parallel and union ----------
        all_chars: set[str] = set()
        for chunk_set in Parallel(n_jobs=n_jobs)(
            delayed(_chars_from_offset_batch)(batch)
            for batch in tqdm(_chunk(offsets, batch_size))
        ):
            all_chars.update(chunk_set)

        lg.info("Fen Builder Finish")

        return sorted(all_chars)
        
    @classmethod
    def _builder_move(cls) -> List[str]:
        lg.info("Move Builder Start")
        all_uci = set()

        # 1) Enumerate base moves for each piece from each square on an empty board
        #    Use pseudo-legal to avoid king requirements.
        piece_types = [
            chess.PAWN, chess.KNIGHT, chess.BISHOP,
            chess.ROOK, chess.QUEEN, chess.KING
        ]
        for square in chess.SQUARES:
            for pt in piece_types:
                b = chess.Board.empty()
                b.turn = chess.WHITE  # color doesn't matter for UCI coverage
                b.set_piece_at(square, chess.Piece(pt, chess.WHITE))
                for mv in b.pseudo_legal_moves:
                    all_uci.add(mv.uci())

        # 2) Explicitly add ALL promotion variants (both colors, capture & non-capture)
        #    White: rank 6 -> 7  (e7e8*)
        #    Black: rank 1 -> 0  (e2e1*)
        promo_pieces = (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT)
        for color, r_from, r_to in (
            (chess.WHITE, 6, 7),
            (chess.BLACK, 1, 0),
        ):
            for f in range(8):
                from_sq = chess.square(f, r_from)

                # Non-capture promotions (straight push): add q, r, b, n
                to_sq = chess.square(f, r_to)
                for promo in promo_pieces:
                    all_uci.add(chess.Move(from_sq, to_sq, promotion=promo).uci())

                # Capture promotions to the left/right: add q, r, b, n
                for df in (-1, 1):
                    f_to = f + df
                    if 0 <= f_to < 8:
                        to_sq = chess.square(f_to, r_to)
                        for promo in promo_pieces:
                            all_uci.add(chess.Move(from_sq, to_sq, promotion=promo).uci())

        # 3) Add castling UCIs explicitly (UCI uses king moves)
        #    e1g1, e1c1, e8g8, e8c8
        for u in ("e1g1", "e1c1", "e8g8", "e8c8"):
            all_uci.add(u)

        lg.info("Move Builder Finish")
        return sorted(all_uci)

    def __init__(self, pgn_source, force_new = False):
        lg.info("Tokenizer Initialized")
        if not os.path.isfile("./data/tokenizer.parquet") or force_new:
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

            df = pl.DataFrame(vocab, schema={"idx": pl.Int16, "tokens": pl.String})
            df.write_parquet("./data/tokenizer.parquet")

    def encode(self, tokens: str, save_at="./data/tokenizer.parquet") -> List[int]:
        """Supports both fen string and move string"""
        # lg.debug(f"Encoder Saved at: {save_at}")
        df = pl.read_parquet(save_at)

        # fen string
        if len(tokens) > 7:
            token_ids: List[int] = []
            tokens = self._pad_fen(tokens)
            for token in tokens:
                df_filter = df.filter(pl.col("tokens") == token)
                assert df_filter.height == 1, f"Failed for token: {token}, response: {df_filter}"

                token_ids.append(df_filter["idx"].item())

            return token_ids
    
        #move is considered one token
        token = tokens
        df_filter = df.filter(pl.col("tokens") == token)
        assert df_filter.height == 1, f"Failed for token: {token}, response: {df_filter}"

        return [df_filter["idx"].item() - 32]

    def decode(self, token_ids: List[int], read_from="./data/tokenizer.parquet") -> str:
        lg.debug(f"Decoder Read_from: {read_from}")
        df = pl.read_parquet(read_from)
        lg.info(f"Dataframe schema found: {df.schema}")

        tokens: List[str] = []
        for token_id in token_ids:
            df_filter = df.filter(pl.col("idx") == token_id)
            assert df_filter.height == 1

            tokens.append(df_filter["tokens"].item())

        return tokens

if __name__ == "__main__":
    tokenizer = Tokenizer(pgn_source="./lichess_db_standard_rated_2014-07.pgn", force_new = True)

    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    token_ids = tokenizer.encode(fen)
    token = tokenizer.decode(token_ids)
    token = "".join(token)
    print(f"Id: {token_ids}, token: {token}, fen: {fen}")
    print()
    uci = "e2e4"
    token_ids = tokenizer.encode(uci)
    token = tokenizer.decode(token_ids)
    print(f"Id: {token_ids}, token: {token}")
