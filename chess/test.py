from checkpoint import restore_checkpoint_shard
from rich import print
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

from pathlib import PosixPath
import tomllib

import flax.nnx as nnx
from model import Classifier
from train import model_cls_run
from tokenizer import Tokenizer
import chess
import jax
import jax.numpy as jnp
import sys


# Load config and tokenizer
with open("config.toml", "rb") as f:
    cfg = tomllib.load(f)

tokenizer = Tokenizer("")


# Build model configs and checkpoint paths
cfg_small = cfg.copy()
cfg_small["embed_dim"] = 768
cfg_small["ffn_dim"] = 768
cfg_small["layers"] = 12
cfg_small["num_heads"] = 12

chk_path = PosixPath("./checkpoints/checkpoint").absolute()
chk_path_small = chk_path / "cls" / "small" / "4096"
chk_path_xxs = chk_path / "cls" / "xxs" / "4096"


def restore(model, chk_path, last_checkpoint: str):
    try:
        model = restore_checkpoint_shard(model, chk_path, last_checkpoint)
        # print(f"[yellow]Model {last_checkpoint} Restored[/yellow]")
        return model
    except UnboundLocalError:
        print(
            f"[red]last_checkpoint: {last_checkpoint} given in config does not exist[/red]"
        )
        sys.exit(1)


def build_models_for_seed(seed_val: int):
    seed_int = int(seed_val)
    m_small = Classifier(cfg_small, rngs=nnx.Rngs(seed_int))
    m_xxs = Classifier(cfg, rngs=nnx.Rngs(seed_int))
    m_small = restore(m_small, chk_path_small, "4096")
    m_xxs = restore(m_xxs, chk_path_xxs, "4096")
    return m_small, m_xxs


def pick_legal_move(model, board: chess.Board, tokenizer: Tokenizer) -> chess.Move:
    """Return a legal move chosen by the model.

    If the top-argmax move is illegal, iteratively mask it in logits
    and take the next argmax until a legal move is found.
    """
    fen_ids = tokenizer.encode(board.fen())
    inp = jnp.asarray(fen_ids)[None, :]

    logits = model_cls_run(model, inp)
    logits = jnp.asarray(logits)
    if logits.ndim > 1:
        logits = logits[0]

    legal_uci = {m.uci() for m in board.legal_moves}

    tried = 0
    vocab_size = int(logits.shape[0])
    while tried < vocab_size:
        idx = int(jnp.argmax(logits))
        uci = tokenizer.decode(idx)
        if uci in legal_uci:
            return chess.Move.from_uci(uci)
        logits = logits.at[idx].set(-jnp.inf)  # mask invalid
        tried += 1

    raise RuntimeError("No legal move found; position likely terminal.")


def play_game(model_white, model_black, tokenizer: Tokenizer, max_plies: int = 400) -> str:
    """Play a single game and return '1-0', '0-1', or '1/2-1/2'."""
    board = chess.Board()
    plies = 0
    while not board.is_game_over() and plies < max_plies:
        current = model_white if board.turn == chess.WHITE else model_black
        move = pick_legal_move(current, board, tokenizer)
        board.push(move)
        plies += 1

    if not board.is_game_over():
        return "1/2-1/2"
    return board.result()


def run_matches(tokenizer: Tokenizer, n_games: int = 10):
    """Run n_games alternating colors, initializing fresh models each match.

    For each match, both models are initialized with the same integer seed,
    derived from cfg['seed'] + game_index. Draws are adjudicated randomly
    to ensure a decisive outcome.
    Returns (small_wins, xxs_wins).
    """
    a_wins = b_wins = 0
    try:
        base_seed = int(cfg.get("seed", 0))
    except (ValueError, TypeError):
        print("[red]cfg['seed'] must be an integer[/red]")
        sys.exit(1)

    print(f"[green]Starting {n_games} matches (no-draw mode)...[/green]")
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("•"),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("Preparing...", total=n_games)

        for g in range(n_games):
            progress.update(task, description=f"Game {g+1}/{n_games}")
            seed = base_seed + g
            a_white = (g % 2 == 0)
            print(
                f"[cyan]Game {g+1}/{n_games}[/cyan] seed={seed} | small plays {'White' if a_white else 'Black'}"
            )

            model_a, model_b = build_models_for_seed(seed)
            res = play_game(model_a, model_b, tokenizer) if a_white else play_game(model_b, model_a, tokenizer)

            if res == "1-0":
                if a_white:
                    a_wins += 1
                else:
                    b_wins += 1
            elif res == "0-1":
                if a_white:
                    b_wins += 1
                else:
                    a_wins += 1
            else:
                # Adjudicate draw randomly to avoid ties
                key = jax.random.PRNGKey(seed + 42)
                winner_small = bool(jax.random.bernoulli(key).item())
                if winner_small:
                    a_wins += 1
                    print("Draw adjudicated: model_small awarded win")
                    res = "1-0" if a_white else "0-1"
                else:
                    b_wins += 1
                    print("Draw adjudicated: model_xxs awarded win")
                    res = "0-1" if a_white else "1-0"

            print(f"Result: {res} | score small {a_wins} - xxs {b_wins}")
            progress.advance(task)

    return a_wins, b_wins


if __name__ == "__main__":
    # Run 10 games between models, re-initializing with new seeds each match
    small_wins, xxs_wins = run_matches(tokenizer, n_games=10)

    total = 10
    small_win_pct = 100.0 * small_wins / total
    xxs_win_pct = 100.0 * xxs_wins / total

    print(f"model_small win%: {small_win_pct:.1f}")
    print(f"model_xxs win% : {xxs_win_pct:.1f}")
