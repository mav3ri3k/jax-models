from flax import nnx
import jax
import jax.numpy as jnp

class Transformer(nnx.Module):
    def __init__(self, cfg, *, rngs: nnx.Rngs):
        self.N = cfg['N']
        self.embed     = nnx.Embed(cfg['V'], cfg['D'], rngs=rngs)
        self.pos_embed = nnx.Embed(cfg['L'], cfg['D'], rngs=rngs)

        self.blocks = [TBlock(cfg=cfg, rngs=rngs) for _ in range(self.N)]

        self.out_ln = nnx.RMSNorm(num_features=cfg['D'],  rngs=rngs)


    def __call__(self, x_BL):
        x_BLD = self.embed(x_BL)
        pos_ids = jnp.arange(x_BL.shape[1])[None, :]
        x_BLD += self.pos_embed(pos_ids)

        for i in range(self.N):
                x_BLD = self.blocks[i](x_BLD)
        
        x_BLD = self.out_ln(x_BLD)

        logits_BLV = self.embed.attend(x_BLD)

        return logits_BLV

class TBlock(nnx.Module):
    def __init__(self, cfg, *, rngs=nnx.Rngs):
        self.atn = Attention(cfg=cfg, rngs=rngs)
        self.ffn = Ffn(cfg=cfg, rngs=rngs)
        self.ln1 = nnx.RMSNorm(num_features=cfg['D'], rngs=rngs)
        self.ln2 = nnx.RMSNorm(num_features=cfg['D'], rngs=rngs)

    def __call__(self, x_BLD: jnp.ndarray) -> jnp.ndarray:
        # PaLM-style parallel block: x + Attn(LN1(x)) + FFN(LN2(x))
        a_BLD = self.atn(self.ln1(x_BLD))
        f_BLD = self.ffn(self.ln2(x_BLD))
        return x_BLD + a_BLD + f_BLD

           
class Attention(nnx.Module):
    def __init__(self, cfg, *, rngs: nnx.Rngs):
        self.Dh = cfg['D'] // cfg['H']
        self.q = nnx.LinearGeneral(in_features=cfg['D'], out_features=(cfg['H'], self.Dh), axis=-1, use_bias=False, rngs=rngs)
        self.k = nnx.LinearGeneral(in_features=cfg['D'], out_features=(cfg['H'], self.Dh), axis=-1, use_bias=False, rngs=rngs)
        self.v = nnx.LinearGeneral(in_features=cfg['D'], out_features=(cfg['H'], self.Dh), axis=-1, use_bias=False, rngs=rngs)

        self.out = nnx.LinearGeneral(in_features=(cfg['H'], self.Dh), out_features=cfg['D'], axis=(-2, -1), use_bias=False, rngs=rngs)
    
    def __call__(self, x_BLD: jnp.ndarray) -> jnp.ndarray:
        q_BLHDh = self.q(x_BLD)
        k_BLHDh = self.k(x_BLD)
        v_BLHDh = self.v(x_BLD)

        a_BHLL = jnp.einsum("...qhd,...khd->...hqk", q_BLHDh, k_BLHDh)
        a_BHLL /= self.Dh ** 0.5

        L = x_BLD.shape[1]
        mask_11LL = jnp.tril(jnp.ones((1, 1, L, L), dtype=jnp.bool_))
        NEG_INF = jnp.finfo(jnp.float32).min

        a_BHLL = jnp.where(mask_11LL, a_BHLL, NEG_INF)
        a_BHLL = jax.nn.softmax(a_BHLL)
        a_BLHDh = jnp.einsum('...hqk,...khd->...qhd', a_BHLL, v_BLHDh)

        a_BLD = self.out(a_BLHDh)
        return a_BLD

class Ffn(nnx.Module):
    def __init__(self, cfg, *, rngs: nnx.Rngs):
        self.gate = nnx.Linear(in_features=cfg['D'], out_features=cfg['Dh'], use_bias=False, rngs=rngs) 
        self.value = nnx.Linear(in_features=cfg['D'], out_features=cfg['Dh'], use_bias=False, rngs=rngs) 
        self.out = nnx.Linear(in_features=cfg['Dh'], out_features=cfg['D'], use_bias=False, rngs=rngs) 

    def __call__(self, x_BLD: jnp.ndarray) -> jnp.ndarray:
        x_BLF = self.gate(x_BLD)
        x_BLF = nnx.swish(x_BLF)

        y_BLF = self.value(x_BLD)

        y_BLF *= x_BLF        
        z_BLD = self.out(y_BLF)

        return z_BLD

