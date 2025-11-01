import jax
import jax.numpy as jnp

def sample(key, logits, temperature: float = 1.0, top_p: float | None = None):
    """Temperature + optional top-p (nucleus) sampling.

    - Greedy if temperature <= 0
    - If top_p is None or >= 1.0, plain categorical
    - Else, restrict to smallest set with cumulative prob >= top_p
    """
    if temperature <= 0:
        return jnp.argmax(logits, axis=-1)

    # Temperature scaling
    scaled = logits / jnp.maximum(temperature, 1e-8)

    # Plain categorical if no top-p
    if (top_p is None) or (top_p >= 1.0):
        return jax.random.categorical(key, logits=scaled, axis=-1)

    # Sort tokens by descending logit
    sorted_idx = jnp.argsort(scaled, axis=-1)[..., ::-1]
    sorted_logits = jnp.take_along_axis(scaled, sorted_idx, axis=-1)

    # Compute probabilities and cumulative mass
    sorted_log_probs = jax.nn.log_softmax(sorted_logits, axis=-1)
    sorted_probs = jnp.exp(sorted_log_probs)
    cdf = jnp.cumsum(sorted_probs, axis=-1)

    # Keep tokens while cumulative mass BEFORE the token is < top_p
    keep = (cdf - sorted_probs) < top_p
    keep = keep.at[..., 0].set(True)  # always keep the highest-prob token

    # Mask out tokens beyond the nucleus and map back to original index order
    masked_sorted_logits = jnp.where(keep, sorted_logits, -jnp.inf)
    inv_idx = jnp.argsort(sorted_idx, axis=-1)
    masked_logits = jnp.take_along_axis(masked_sorted_logits, inv_idx, axis=-1)

    return jax.random.categorical(key, logits=masked_logits, axis=-1)


def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    cfg,
    seed: int | None = None,
):
    # 1. Tokenize prompt
    token_ids = tokenizer.encode(prompt)
    ids = jnp.array([token_ids], dtype=jnp.int32)  # shape: (1, len)

    # Initialize RNG
    key = jax.random.PRNGKey(cfg['seed'] if seed is None else seed)

    for _ in range(max_new_tokens):
        # 2. Truncate to max context length
        ids_cond = ids[:, -cfg['L']:]  # causal model only uses last L tokens

        # 3. Predict next token logits
        logits = model(ids_cond)  # shape: (1, seq_len, vocab)
        next_token_logits = logits[:, -1, :]  # only the last token's output

        # 4. Pick next token
        key, subkey = jax.random.split(key)
        next_token = sample(subkey, next_token_logits, cfg['tmp'], top_p=cfg['p'])

        # 5. Append
        ids = jnp.concatenate([ids, next_token[:, None]], axis=-1)

    return tokenizer.decode(ids[0].tolist())

def chat(model, tokenizer, max_new_tokens: int, cfg):
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt
    from rich import box

    console = Console()
    console.print(
        Panel(
            "Interactive chat. Type [bold]/exit[/] to quit.",
            title="attoBabble Chat",
            border_style="bright_blue",
            box=box.ROUNDED,
        )
    )

    try:
        while True:
            user_input = Prompt.ask("[bold magenta]You[/]", default="", show_default=False).strip()

            if user_input.lower() == "/exit":
                console.print("[dim]Exiting chat. Goodbye![/dim]")
                break

            if not user_input:
                continue

            with console.status("[bold cyan]Generating...[/]", spinner="dots"):
                output = generate(model, tokenizer, user_input, max_new_tokens, cfg)

            console.print(
                Panel.fit(
                    output,
                    title="Assistant",
                    border_style="cyan",
                    box=box.ROUNDED,
                )
            )
    except KeyboardInterrupt:
        console.print("\n[dim]Chat interrupted. Bye![/dim]")
