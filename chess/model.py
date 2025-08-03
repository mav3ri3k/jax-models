from flax import nnx
import jax
import jax.numpy as jnp

class Ffn(nnx.Module):
    def __init__(self, cfg, *, rngs: nnx.Rngs):
        self.gate = nnx.Linear(in_features=cfg['embed_dim'], out_features=cfg['ffn_dim'], use_bias=cfg['use_bias'], rngs=rngs) 
        self.value = nnx.Linear(in_features=cfg['embed_dim'], out_features=cfg['ffn_dim'], use_bias=cfg['use_bias'], rngs=rngs) 
        self.out = nnx.Linear(in_features=cfg['ffn_dim'], out_features=cfg['embed_dim'], use_bias=cfg['use_bias'], rngs=rngs) 

    def __call__(self, x_BLD: jnp.ndarray) -> jnp.ndarray:
        x_BLF = self.gate(x_BLD)
        x_BLF = nnx.swish(x_BLF)

        y_BLF = self.value(x_BLD)

        y_BLF *= x_BLF        
        z_BLD = self.out(y_BLF)

        return z_BLD

class Attention(nnx.Module):
    def __init__(self, cfg, *, rngs: nnx.Rngs):
        assert cfg['embed_dim'] % cfg['num_heads'] == 0, f"{embed_dim} and {num_heads} are not divisible"
        self.Dh = cfg['embed_dim'] // cfg['num_heads']
        self.q = nnx.LinearGeneral(in_features=cfg['embed_dim'], out_features=(cfg['num_heads'], self.Dh), axis=-1, use_bias=cfg['use_bias'], rngs=rngs)
        self.k = nnx.LinearGeneral(in_features=cfg['embed_dim'], out_features=(cfg['num_heads'], self.Dh), axis=-1, use_bias=cfg['use_bias'], rngs=rngs)
        self.v = nnx.LinearGeneral(in_features=cfg['embed_dim'], out_features=(cfg['num_heads'], self.Dh), axis=-1, use_bias=cfg['use_bias'], rngs=rngs)

        self.normq = nnx.RMSNorm(num_features=self.Dh, rngs=rngs)
        self.normk = nnx.RMSNorm(num_features=self.Dh, rngs=rngs)

        self.out = nnx.LinearGeneral(in_features=(cfg['num_heads'], self.Dh), out_features=cfg['embed_dim'], axis=(-2, -1), use_bias=cfg['use_bias'], rngs=rngs)

    
    def __call__(self, x_BLD: jnp.ndarray) -> jnp.ndarray:
        q_BLHDh = self.q(x_BLD)
        q_BLHDh = self.normq(q_BLHDh)

        k_BLHDh = self.k(x_BLD)
        k_BLHDh = self.normk(k_BLHDh)

        v_BLHDh = self.v(x_BLD)

        a_BLHDh = nnx.dot_product_attention(q_BLHDh, k_BLHDh, v_BLHDh)

        a_BLD = self.out(a_BLHDh)
        return a_BLD

class TBlock(nnx.Module):
    def __init__(self, cfg, *, rngs: nnx.Rngs):
        self.atn = Attention(cfg=cfg, rngs=rngs)
        self.ffn = Ffn(cfg=cfg, rngs=rngs)
        self.ln1 = nnx.RMSNorm(num_features=cfg['embed_dim'], rngs=rngs)
        self.ln2 = nnx.RMSNorm(num_features=cfg['embed_dim'], rngs=rngs)

    def __call__(self, in_BLD: jnp.ndarray) -> jnp.ndarray:
        x_BLD = self.ln1(in_BLD)
        x_BLD = self.atn(x_BLD)
        x_BLD += in_BLD

        y_BLD = self.ln2(x_BLD)
        y_BLD = self.ffn(y_BLD) 
        y_BLD += x_BLD 

        return y_BLD


class Encoder(nnx.Module):
    def __init__(self, cfg, *, rngs: nnx.Rngs):
        self.N = cfg['layers']

        pos_emb_shape = (1, cfg['ctx_len'], cfg['embed_dim'])
        self.pos_embed = nnx.Param(jax.nn.initializers.normal(0.02)(jax.random.key(cfg['seed']), pos_emb_shape, jnp.float32))

        self.blocks = [TBlock(cfg=cfg, rngs=rngs) for _ in range(self.N)]

        self.out_ln = nnx.RMSNorm(num_features=cfg['embed_dim'], use_scale=False, rngs=rngs)


    def __call__(self, x_BLD):
        # learned pos embedding
        x_BLD += self.pos_embed

        for i in range(self.N):
                x_BLD = self.blocks[i](x_BLD)
        
        x_BLD = self.out_ln(x_BLD)

        return x_BLD

class VisionTransformer(nnx.Module):
    def __init__(self, cfg, *, rngs: nnx.Rngs):
        self.embed = nnx.Embed(num_embeddings=cfg['vocab_size'], features=cfg['embed_dim'], rngs=rngs)
        self.class_token = nnx.Param(jax.nn.initializers.zeros(jax.random.key(cfg['seed']), (1, 1, cfg['embed_dim']), jnp.float32))

        self.encoder = Encoder(cfg, rngs=rngs)

        self.out = nnx.Linear(in_features=cfg['embed_dim'], out_features=cfg['num_classes'], rngs=rngs)


    def __call__(self, x_BL):
        # We can merge s2d+emb into a single conv; it's the same.

        x_BLD = self.embed(x_BL)

        # Add [class] token
        cls = jnp.tile(self.class_token, [x_BLD.shape[0], 1, 1])  
        x_BLD = jnp.concatenate([cls, x_BLD], axis=1)

        # main transformer encoder
        x_BLD = self.encoder(x_BLD)

        # extract [class] token
        x_BD = x_BLD[:, 0]

        # find class
        x_BC = self.out(x_BD)
        
        return x_BC

