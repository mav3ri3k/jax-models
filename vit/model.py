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

class TBlock(nnx.Module):
    def __init__(self, cfg, *, rngs: nnx.Rngs):
        self.atn = nnx.MultiHeadAttention(num_heads=cfg['num_heads'], in_features=cfg['embed_dim'], decode=False, rngs=rngs)
        self.ffn = Ffn(cfg=cfg, rngs=rngs)
        self.ln1 = nnx.LayerNorm(num_features=cfg['embed_dim'], use_bias=cfg['use_bias'], rngs=rngs)
        self.ln2 = nnx.LayerNorm(num_features=cfg['embed_dim'], use_bias=cfg['use_bias'], rngs=rngs)

    def __call__(self, in_BLD: jnp.ndarray) -> jnp.ndarray:
        # order of layernorm based on palm
        # parallel formulation
        x_BLD = self.ln1(in_BLD)
        x_BLD = self.atn(x_BLD)

        y_BLD = self.ln2(in_BLD)
        x_BLD += y_BLD 

        z_BLD = self.ffn(x_BLD) 
        z_BLD += x_BLD

        return z_BLD


class Encoder(nnx.Module):
    def __init__(self, cfg, *, rngs: nnx.Rngs):
        self.N = cfg['layers']

        i = cfg['image_size']
        p = cfg['patch_size']
        pos_emb_shape = (1, int((i/p)*(i/p)) + 1, cfg['embed_dim'])
        self.pos_embed = nnx.Param(jax.nn.initializers.normal(0.02)(jax.random.key(cfg['seed']), pos_emb_shape, jnp.float32))

        self.blocks = [TBlock(cfg=cfg, rngs=rngs) for _ in range(self.N)]

        self.out_ln = nnx.LayerNorm(num_features=cfg['embed_dim'], use_bias=cfg['use_bias'], use_scale=False, rngs=rngs)


    def __call__(self, x_BLD):
        # learned pos embedding
        x_BLD += self.pos_embed

        for i in range(self.N):
                x_BLD = self.blocks[i](x_BLD)
        
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

