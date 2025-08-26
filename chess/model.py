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
    @staticmethod
    def rotary_angles(cfg):
        """Return cos and sin tables shaped (seq_len, d_head//2)."""
        # Compute inverse frequencies   (d_head // 2,)  – one per 2-D pair
        dh = cfg['embed_dim']//cfg['num_heads']

        idx = jnp.arange(0, dh, 2)
        inv_freq = 1.0 / (cfg['base_f'] ** (idx / dh))
        # Positions                                    (seq_len, 1)
        pos = jnp.arange(cfg['ctx_len'])[:, None]
        # Angles θ_{i,j}                               (seq_len, d_head//2)
        angles = pos * inv_freq
        cos = jnp.cos(angles)
        sin = jnp.sin(angles)

        # Reshape for broadcasting: (1, seq, 1, d_head//2)
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]

        return cos, sin

    @staticmethod
    def apply_rope(x_BLHDh, cos, sin):
        """
        x        – (batch, seq, n_heads, d_head)
        cos/sin  – (seq, d_head//2)      broadcast on batch & heads
        returns  – same shape as x, with RoPE applied
        """
        # Split the head dimension into even/odd pairs
        e_BLHDh = x_BLHDh[..., 0::2]           # (b, s, h, d/2)
        o_BLHDh  = x_BLHDh[..., 1::2]

        # Rotate:   (x_even, x_odd) → (x_even*cos − x_odd*sin, x_even*sin + x_odd*cos)
        re_BLHDh = e_BLHDh * cos - o_BLHDh * sin
        ro_BLHDh  = e_BLHDh * sin + o_BLHDh * cos

        # Re-interleave the last dimension
        
        x = jnp.stack([re_BLHDh, ro_BLHDh], axis=-1)

        return jnp.reshape(x, x_BLHDh.shape)

    def __init__(self, cfg, *, rngs: nnx.Rngs):
        assert cfg['embed_dim'] % cfg['num_heads'] == 0, f"{embed_dim} and {num_heads} are not divisible"
        self.Dh = cfg['embed_dim'] // cfg['num_heads']
        self.q = nnx.LinearGeneral(in_features=cfg['embed_dim'], out_features=(cfg['num_heads'], self.Dh), axis=-1, use_bias=cfg['use_bias'], rngs=rngs)
        self.k = nnx.LinearGeneral(in_features=cfg['embed_dim'], out_features=(cfg['num_heads'], self.Dh), axis=-1, use_bias=cfg['use_bias'], rngs=rngs)
        self.v = nnx.LinearGeneral(in_features=cfg['embed_dim'], out_features=(cfg['num_heads'], self.Dh), axis=-1, use_bias=cfg['use_bias'], rngs=rngs)

        self.normq = nnx.RMSNorm(num_features=self.Dh, rngs=rngs)
        self.normk = nnx.RMSNorm(num_features=self.Dh, rngs=rngs)

        cos, sin = self.rotary_angles(cfg)
        self.cos, self.sin = nnx.Variable(cos, mutable=False), nnx.Variable(sin, mutable=False)

        self.out = nnx.LinearGeneral(in_features=(cfg['num_heads'], self.Dh), out_features=cfg['embed_dim'], axis=(-2, -1), use_bias=cfg['use_bias'], rngs=rngs)

    
    def __call__(self, x_BLD: jnp.ndarray) -> jnp.ndarray:
        q_BLHDh = self.q(x_BLD)
        q_BLHDh = self.normq(q_BLHDh)
        q_BLHDh = self.apply_rope(q_BLHDh, self.cos, self.sin)

        k_BLHDh = self.k(x_BLD)
        k_BLHDh = self.normk(k_BLHDh)
        k_BLHDh = self.apply_rope(k_BLHDh, self.cos, self.sin)

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


class Transformer(nnx.Module):
    def __init__(self, cfg, *, rngs: nnx.Rngs):
        self.N = cfg['layers']

        pos_emb_shape = (1, cfg['ctx_len'], cfg['embed_dim'])
        self.blocks = [TBlock(cfg=cfg, rngs=rngs) for _ in range(self.N)]

        self.out_ln = nnx.RMSNorm(num_features=cfg['embed_dim'], use_scale=False, rngs=rngs)

    def __call__(self, x_BLD):
        for i in range(self.N):
                x_BLD = self.blocks[i](x_BLD)
        
        x_BLD = self.out_ln(x_BLD)

        return x_BLD


class Classifier(nnx.Module):

    def __init__(self, cfg, *, rngs: nnx.Rngs):
        self.embed = nnx.Embed(num_embeddings=cfg['vocab_size'], features=cfg['embed_dim'], rngs=rngs)
        self.class_token = nnx.Param(jax.nn.initializers.zeros(jax.random.key(cfg['seed']), (1, 1, cfg['embed_dim']), jnp.float32))

        self.encoder = Transformer(cfg, rngs=rngs)

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

class EBM(nnx.Module):
    def __init__(self, cfg, *, rngs: nnx.Rngs):
        self.board_embed = nnx.Embed(num_embeddings=cfg['vocab_size'], features=cfg['embed_dim'], rngs=rngs)

        self.encoder = Transformer(cfg, rngs=rngs)
        self.out = nnx.LinearGeneral(in_features=(cfg['ctx_len'], cfg['embed_dim']), out_features=1, axis=(-2, -1), rngs=rngs)

    def __call__(self, b_BL, m_BD):
        # embedding
        b_BLD = self.board_embed(b_BL)
        m_BLD = m_BD[:, None, :]

        b_BLD = jnp.concatenate([m_BLD, b_BLD], axis=1)
        b_BLD = self.encoder(b_BLD)

        e_BE = self.out(b_BLD)

        # e_B = jnp.squeeze(e_BE, axis=-1)
        
        return e_BE

class EBMChess(nnx.Module):
    def __init__(self, cfg, *, rngs: nnx.Rngs):
        self.move_embed = nnx.Embed(num_embeddings=cfg['num_classes'], features=cfg['embed_dim'], rngs=rngs)
        self.rngs = rngs
        self.embed_dim = cfg['embed_dim']
        self.steps = cfg['steps']
        self.alpha = cfg['alpha']
        self.sigma = cfg['sigma']

        self.energy_model = EBM(cfg, rngs=rngs)
        def energy_fn(b_BL, m_BD):
            return jnp.sum(self.energy_model(b_BL, m_BD))

        self.energy_grad = nnx.grad(energy_fn, argnums=1)

    def __call__(self, b_BL):
        m_BD = jax.random.normal(self.rngs.default(), (b_BL.shape[0], self.embed_dim), dtype=jnp.float32)

        for _ in range(self.steps):
            m_BD = jax.nn.softmax(m_BD)
            grads = self.energy_grad(b_BL, m_BD)
            m_BD = m_BD - self.alpha * grads
            m_BD += self.sigma * jax.random.normal(self.rngs.default(), m_BD.shape)

        m_BM =  self.move_embed.attend(m_BD.astype(jnp.float32))

        return m_BM
