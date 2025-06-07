import sentencepiece as spm
import tomllib
import jax
import jax.numpy as jnp

def download_shakespeare():
    import requests

    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    response = requests.get(url)
    
    if response.status_code == 200:
        with open("input.txt", "w", encoding='utf-8') as f:
            f.write(response.text)
        print("✅ Downloaded and saved as input.txt")
    else:
        print(f"❌ Failed to download. Status code: {response.status_code}")

def train_tokenizer():    
    with open("config.toml", "rb") as f:
        cfg = tomllib.load(f)

    spm.SentencePieceTrainer.train(
        input='./input.txt',
        model_prefix='attoBabble',
        vocab_size=cfg['V'],
        model_type='bpe',
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3
    )

def get_tokenizer():
    """
    text = "Hello, how are you today?"
    ids = get_tokenizer().encode(text, out_type=int)
    reconstructed = sp.decode(ids)
    """
    sp = spm.SentencePieceProcessor()
    sp.load("attoBabble.model")

    return sp

def load_and_tokenize(path: str, seq_len: int) -> jnp.ndarray:
    """
    Reads `path` into one long token list, then chops it into
    blocks of length (seq_len + 1), returned as a [n_blocks, seq_len+1] array.
    """
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()

    tokenizer = get_tokenizer()
    token_ids = tokenizer.encode(text)
    ids = jnp.array(token_ids, dtype=jnp.int32)  # [total_len]

    block_size = seq_len + 1
    total_len = ids.shape[0]
    n_blocks = total_len // block_size
    if n_blocks == 0:
        raise ValueError(f"Not enough tokens ({total_len}) for one block of size {block_size}")

    usable_len = n_blocks * block_size
    ids = ids[:usable_len]                        # [usable_len]

    return ids.reshape((n_blocks, block_size))    # [n_blocks, seq_len+1]

def batch_iterator(
    ids: jnp.ndarray,
    batch_size: int,
    shuffle: bool = True,
    seed: int = 0
):
    """
    Yields tuples (x, y) each of shape [batch_size, seq_len]:
      x = ids[:, :-1]
      y = ids[:,  1:]
    """
    # 1) Shuffle at the block level
    if shuffle:
        key = jax.random.PRNGKey(seed)
        perm = jax.random.permutation(key, ids.shape[0])
        ids = ids[perm]

    # 2) Split into inputs & next-token targets
    x = ids[:, :-1]   # [n_blocks, seq_len]
    y = ids[:,  1:]   # [n_blocks, seq_len]

    # 3) Yield fixed-size batches
    n_batches = x.shape[0] // batch_size
    for i in range(n_batches):
        xb = x[i*batch_size : (i+1)*batch_size]
        yb = y[i*batch_size : (i+1)*batch_size]
        yield xb, yb
