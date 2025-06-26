import math
from flax import nnx
import jax
import jax.numpy as jnp

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
        self.ln1 = nnx.LayerNorm(num_features=cfg['embed_dim'], use_bias=False, use_scale=False, rngs=rngs)
        self.ln2 = nnx.LayerNorm(num_features=cfg['embed_dim'], use_bias=False, use_scale=False, rngs=rngs)
        self.adn = AdaLN_Zero(cfg=cfg, rngs=rngs)

    def __call__(self, in_BLD: jnp.ndarray, c_BD: jnp.ndarray) -> jnp.ndarray:
        a1_BD, b1_BD, c1_BD, a2_BD, b2_BD, c2_BD = jnp.split(self.adn(c_BD), 6, axis=-1)

        x_BLD = self.ln1(in_BLD)
        x_BLD = (1 + a1_BD[:, None, :]) * x_BLD + b1_BD[:, None, :]
        x_BLD = self.atn(x_BLD)
        x_BLD *= c1_BD[:, None, :]
        x_BLD += in_BLD

        y_BLD = self.ln2(x_BLD)
        y_BLD = (1 + a2_BD[:, None, :]) * y_BLD + b2_BD[:, None, :]
        y_BLD = self.ffn(y_BLD) 
        y_BLD *= c2_BD[:, None, :]
        y_BLD += x_BLD 

        return y_BLD

class Decoder(nnx.Module):
    def __init__(self, cfg, *, rngs: nnx.Rngs):
        self.N = cfg['layers']
        self.blocks = [TBlock(cfg=cfg, rngs=rngs) for _ in range(self.N)]

    def __call__(self, x_BLD: jnp.ndarray, c_BD: jnp.ndarray) -> jnp.ndarray:
        for i in range(self.N):
                x_BLD = self.blocks[i](x_BLD, c_BD)
        
        return x_BLD

class FinalLayer(nnx.Module):
    def __init__(self, cfg, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(in_features=cfg['embed_dim'], out_features=2 * cfg['embed_dim'], use_bias=cfg['use_bias'], rngs=rngs) 
        self.ln = nnx.LayerNorm(num_features=cfg['embed_dim'], use_bias=False, use_scale=False, rngs=rngs)
        self.out = nnx.Linear(in_features=cfg['embed_dim'], out_features=cfg['patch_size']**2 * cfg['channels'], use_bias=cfg['use_bias'], rngs=rngs) 

        self.b = cfg['batch_size']
        self.c = cfg['channels']
        self.p = cfg['patch_size']
        self.h = cfg['image_size'] // self.p

    def __call__(self, x_BLD: jnp.ndarray, c_BD: jnp.ndarray) -> jnp.ndarray:
        
        # adaln zero in final layer
        c_BD = nnx.silu(c_BD)
        c_B2D = self.linear(c_BD)
        a1_BD, b1_BD = jnp.split(c_B2D, 2, axis=-1)
        x_BLD = self.ln(x_BLD)
        x_BLD = (1 + a1_BD[:, None, :]) * x_BLD + b1_BD[:, None, :]

        # linear to get shape like image
        x_BLD = self.out(x_BLD)

        # reshape to image
        x_BhwPPC = jnp.reshape(x_BLD, (self.b, self.h, self.h, self.p, self.p, self.c))
        x_BhPwPC = jnp.einsum('bhwpqc->bhpwqc', x_BhwPPC)
        x_BHWC = jnp.reshape(x_BhPwPC, (self.b, self.h * self.p, self.h * self.p, self.c))
        
        return x_BHWC

class ConditionEmbeder(nnx.Module):
    def __init__(self, cfg, *, rngs: nnx.Rngs):
        self.label_embeddings = nnx.Embed(num_embeddings=cfg['labels'], features=cfg['embed_dim'], rngs=rngs)

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

    def __call__(self, l_B: jnp.ndarray, t_B: jnp.ndarray) -> jnp.ndarray:
        l_BD = self.label_embeddings(l_B) 

        t_BD = self.timestep_embedding(t_B)
        t_BD = self.linear1(t_BD)
        t_BD = nnx.silu(t_BD)
        t_BD = self.linear2(t_BD)

        return l_BD + t_BD
    
class VisionTransformer(nnx.Module):
    def __init__(self, cfg, *, rngs: nnx.Rngs):
        i = cfg['image_size']
        p = cfg['patch_size']
        pos_emb_shape = (1, int((i/p)*(i/p)), cfg['embed_dim'])
        self.pos_embed = nnx.Param(jax.nn.initializers.normal(0.02)(jax.random.key(cfg['seed']), pos_emb_shape, jnp.float32))

        self.cnd_embed = ConditionEmbeder(cfg=cfg, rngs=rngs)

        patch = (p, p)

        self.conv = nnx.Conv(in_features=cfg['channels'], out_features=cfg['embed_dim'], kernel_size=patch, strides=patch, padding='VALID', rngs=rngs)
        self.class_token = nnx.Param(jax.nn.initializers.zeros(jax.random.key(cfg['seed']), (1, 1, cfg['embed_dim']), jnp.float32))

        self.decoder = Decoder(cfg, rngs=rngs)
        self.out = FinalLayer(cfg, rngs=rngs)

    def __call__(self, x_BHWC: jnp.ndarray, t_B: jnp.ndarray, l_B: jnp.ndarray) -> jnp.ndarray:
        # We can merge s2d+emb into a single conv; it's the same.
        x_BPPD = self.conv(x_BHWC)
        b, h, w, d = x_BPPD.shape
        x_BLD = jnp.reshape(x_BPPD, [b, h*w, d])

        c_BD = self.cnd_embed(l_B, t_B)

        # learned pos embedding
        x_BLD += self.pos_embed

        # main transformer decoder
        x_BLD = self.decoder(x_BLD, c_BD)

        # final layer
        x_BHWC = self.out(x_BLD, c_BD)

        return x_BHWC

