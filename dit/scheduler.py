import jax.numpy as jnp
import flax.nnx as nnx

class GaussianDiffusion(nnx.Module):
    def __init__(self, cfg, *, rngs: nnx.Rngs):
        betas = jnp.linspace(cfg['beta_0'], cfg['beta_t'], cfg['diff_steps'], dtype=jnp.float64)
        alphas = 1 - betas
        self.alphas_cumprod = jnp.cumprod(alphas)

        eps = jax.random.normal(noise_key, images.shape)

    def __call__(self, x_BHWC: jnp.ndarray, t_B, eps_B) -> jnp.ndarray:
        alpha_prime = self.alphas_cumprod[t_B]

        xt = jnp.sqrt(alpha_prime) * x_BHWC + jnp.sqrt(1 - alpha_prime) * eps_B
