import jax
import jax.numpy as jnp
from jax.tree_util import Partial as jax_partial

from .pdmp import PDMP





class Boomerang(PDMP):
    def __init__(
        self,
        dim,
        grad_U,
        grid_size=100,
        tmax=2.0,
        refresh_rate=0.1,
        signed_bound=True,
        adaptive=True,
        **kwargs,
    ):
        self.dim = dim
        self.refresh_rate = refresh_rate
        self.true_grad_U = jax_partial(grad_U)
        self.grad_U = jax_partial(lambda x: grad_U(x) - x)
        self.grid_size = grid_size
        if tmax == 0:
            self.tmax = 1.0
            self.adaptive = True
        else:
            self.tmax = float(tmax)
            self.adaptive = adaptive

        self.vectorized_bound = False
        self.signed_bound = signed_bound

        def _integrator_boomerang(x, v, t):
            xt = x * jnp.cos(t) + v * jnp.sin(t)
            vt = -x * jnp.sin(t) + v * jnp.cos(t)
            return xt, vt

        self.integrator = jax_partial(_integrator_boomerang)

        self.rate, self.rate_vect, self.signed_rate, self.signed_rate_vect = (
            self._init_bps_rate()
        )

        def _velocity_jump(x, v, key):

            grad_U_x = self.grad_U(x)
            dim = x.shape[0]
            bounce_prob = jnp.maximum(0.0, grad_U_x @ v)
            bounce_prob /= bounce_prob + self.refresh_rate
            subkey, subkey2 = jax.random.split(key, 2)
            u = jax.random.uniform(subkey)

            def _reflect(vect, normal):
                normal = normal / jnp.linalg.norm(normal)
                return vect - 2 * (vect @ normal) * normal

            out = jnp.where(
                u < bounce_prob,
                _reflect(v, grad_U_x),
                jax.random.normal(subkey2, shape=(dim,)),
            )
            return out

        self.velocity_jump = jax_partial(_velocity_jump)
        self.state = None
