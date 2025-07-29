from flax import nnx
import jax
import jax.numpy as jnp
import math

class Ffn(nnx.Module):
    def __init__(self, cfg, *, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(in_features=cfg['embed_dim'], out_features=cfg['ffn_dim'], use_bias=cfg['use_bias'], rngs=rngs) 
        self.linear2 = nnx.Linear(in_features=cfg['ffn_dim'], out_features=cfg['embed_dim'], use_bias=cfg['use_bias'], rngs=rngs) 

    def __call__(self, x_BLD: jnp.ndarray) -> jnp.ndarray:
        x_BLF = self.linear1(x_BLD)
        x_BLF = nnx.gelu(x_BLF)
        x_BLD = self.linear2(x_BLF)

        return x_BLD

class AdaLN_Zero(nnx.Module):
    def __init__(self, cfg, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(in_features=cfg['embed_dim'], out_features=6 * cfg['embed_dim'], use_bias=cfg['use_bias'], rngs=rngs) 

    def __call__(self, x_BD: jnp.ndarray) -> jnp.ndarray:
        x_BD = nnx.silu(x_BD)
        x_B6D = self.linear(x_BD)

        return x_B6D

class TBlock(nnx.Module):
    def __init__(self, cfg, *, rngs: nnx.Rngs):
        self.atn = nnx.MultiHeadAttention(num_heads=cfg['num_heads'], in_features=cfg['embed_dim'], decode=False, rngs=rngs)
        self.ffn = Ffn(cfg=cfg, rngs=rngs)
        self.ln1 = nnx.LayerNorm(num_features=cfg['embed_dim'], use_bias=cfg['use_bias'], rngs=rngs)
        self.ln2 = nnx.LayerNorm(num_features=cfg['embed_dim'], use_bias=cfg['use_bias'], rngs=rngs)
        self.adn = AdaLN_Zero(cfg=cfg, rngs=rngs)


    def __call__(self, in_BLD: jnp.ndarray, d_D: jnp.ndarray) -> jnp.ndarray:
        a1_D, b1_D, c1_D, a2_D, b2_D, c2_D = jnp.split(self.adn(d_D), 6, axis=-1)

        x_BLD = self.ln1(in_BLD)
        x_BLD = (1 + a1_D[None, None :]) * x_BLD + b1_D[None, None, :]
        x_BLD = self.atn(x_BLD)
        x_BLD *= c1_D[None, None, :]
        x_BLD += in_BLD

        y_BLD = self.ln2(x_BLD)
        y_BLD = (1 + a2_D[None, None, :]) * y_BLD + b2_D[None, None, :]
        y_BLD = self.ffn(y_BLD) 
        y_BLD *= c2_D[None, None, :]
        y_BLD += x_BLD 

        return y_BLD

class ConditionEmbeder(nnx.Module):
    def __init__(self, cfg, *, rngs: nnx.Rngs):
        self.frequency_embedding_size = cfg['embed_dim']
        self.linear1 = nnx.Linear(in_features=cfg['embed_dim'], out_features=cfg['embed_dim'], use_bias=cfg['use_bias'], rngs=rngs) 
        self.linear2 = nnx.Linear(in_features=cfg['embed_dim'], out_features=cfg['embed_dim'], use_bias=cfg['use_bias'], rngs=rngs) 

    # copied, I have to learn this TODO
    def timestep_embedding(self, t, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        t = jax.lax.convert_element_type(t, jnp.float32)
        dim = self.frequency_embedding_size
        half = dim // 2
        freqs = jnp.exp( -math.log(max_period) * jnp.arange(start=0, stop=half, dtype=jnp.float32) / half)
        args = t[:, None] * freqs[None]
        embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)

        return embedding

    def __call__(self,t_B: jnp.ndarray) -> jnp.ndarray:
        t_BD = self.timestep_embedding(t_B)
        t_BD = self.linear1(t_BD)
        t_BD = nnx.silu(t_BD)
        t_BD = self.linear2(t_BD)

        return t_BD

class Encoder(nnx.Module):
    def __init__(self, cfg, *, rngs: nnx.Rngs):
        self.N = cfg['layers']

        i = cfg['image_size']
        p = cfg['patch_size']
        pos_emb_shape = (1, int((i/p)*(i/p)) + 1, cfg['embed_dim'])
        self.pos_embed = nnx.Param(jax.nn.initializers.normal(0.02)(jax.random.key(cfg['seed']), pos_emb_shape, jnp.float32))

        self.block = TBlock(cfg=cfg, rngs=rngs)
        self.depth_embed = ConditionEmbeder(cfg=cfg, rngs=rngs)
        

        self.out_ln = nnx.LayerNorm(num_features=cfg['embed_dim'], use_bias=cfg['use_bias'], use_scale=False, rngs=rngs)


    def __call__(self, x_BLD):
        # learned pos embedding
        x_BLD += self.pos_embed

        d_B = jnp.arange(start=0, stop=self.N)
        d_BD = self.depth_embed(d_B)

        for i in range(self.N):
                x_BLD = self.block(x_BLD, d_BD[i, :])
        
        x_BLD = self.out_ln(x_BLD)

        return x_BLD

class VisionTransformer(nnx.Module):
    def __init__(self, cfg, *, rngs: nnx.Rngs):
        p = cfg['patch_size']
        patch = (p, p)

        self.conv = nnx.Conv(in_features=cfg['channels'], out_features=cfg['embed_dim'], kernel_size=patch, strides=patch, padding='VALID', rngs=rngs)
        self.class_token = nnx.Param(jax.nn.initializers.zeros(jax.random.key(cfg['seed']), (1, 1, cfg['embed_dim']), jnp.float32))

        self.encoder = Encoder(cfg, rngs=rngs)

        self.out = nnx.Linear(in_features=cfg['embed_dim'], out_features=cfg['num_classes'], rngs=rngs)


    def __call__(self, x_BHWC):
        # We can merge s2d+emb into a single conv; it's the same.
        x_BPPD = self.conv(x_BHWC)

        b, h, w, d = x_BPPD.shape
        x_BLD = jnp.reshape(x_BPPD, [b, h*w, d])

        # Add [class] token
        cls = jnp.tile(self.class_token, [b, 1, 1])  
        x_BLD = jnp.concatenate([cls, x_BLD], axis=1)

        # main transformer encoder
        x_BLD = self.encoder(x_BLD)

        # extract [class] token
        x_BD = x_BLD[:, 0]

        # find class, one-hot
        x_BC = self.out(x_BD)
        
        return x_BC

