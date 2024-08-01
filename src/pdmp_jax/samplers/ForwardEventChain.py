import jax
import jax.numpy as jnp
from jax.tree_util import Partial as jax_partial

from typing import  Callable
from jaxtyping import  Array


from .pdmp import PDMP


class ForwardEventChain(PDMP):
    def __init__(
        self,
        dim,
        grad_U,
        grid_size=100,
        tmax=2.0,
        refresh_ortho=0.1,
        signed_bound=True,
        adaptive=True,
        **kwargs,
    ):
        if dim <= 2:
            raise ValueError(
                "The dimension must be greater than 2 to use the ForwardEventChain"
            )
        self.dim = dim
        self.refresh_rate = 0.0  # no refresh rate is used in the forward event chain
        self.refresh_ortho = refresh_ortho
        self.grad_U = jax_partial(grad_U)
        self.grid_size = grid_size
        if tmax == 0:
            self.tmax = 1.0
            self.adaptive = True
        else:
            self.tmax = float(tmax)
            self.adaptive = adaptive
        self.vectorized_bound = False
        self.signed_bound = signed_bound

        self.integrator = jax_partial(lambda x, v, t: (x + (v * t), v))

        self.rate, self.rate_vect, self.signed_rate, self.signed_rate_vect = (
            self._init_bps_rate()
        )

        def _velocity_jump_event_chain(x, v, key):
            subkey1, subkey2, subkey3 = jax.random.split(key, 3)
            dim = x.shape[0]
            u = jax.random.uniform(subkey1)
            rho = -((1 - u ** (2.0 / (dim - 1.0))) ** (0.5))
            grad_U_x = self.grad_U(x)
            grad_U_x = grad_U_x / jnp.linalg.norm(grad_U_x)
            v_par = (v @ grad_U_x) * grad_U_x
            v_ortho = v - v_par

            def _refresh_ortho(key):
                g = jax.random.normal(key, shape=(2, dim))
                g1 = g[0]
                g2 = g[1]
                g1 -= (g1 @ grad_U_x) * grad_U_x
                g2 -= (g2 @ grad_U_x) * grad_U_x
                e1 = g1 / jnp.linalg.norm(g1)
                e2 = g2 - (g2 @ e1) * e1
                e2 = e2 / jnp.linalg.norm(e2)
                theta = jnp.pi / 2
                v_prop = v_ortho - (v_ortho @ e1) * e1 - (v_ortho @ e2) * e2
                v_prop += (jnp.cos(theta) * e1 + jnp.sin(theta) * e2) * (
                    e1 @ v_ortho
                ) + (jnp.sin(theta) * e1 - jnp.cos(theta) * e2) * (e2 @ v_ortho)
                v_prop /= jnp.linalg.norm(v_prop)
                v_prop *= jnp.sign(v_ortho @ v_prop)
                return v_prop

            u2 = jax.random.uniform(subkey2)
            v_prop = jnp.where(
                u2 < self.refresh_ortho,
                _refresh_ortho(subkey3),
                v_ortho / jnp.linalg.norm(v_ortho),
            )
            v_out = v_prop * (1 - rho**2) ** 0.5 + rho * grad_U_x
            return v_out

        self.velocity_jump = jax_partial(_velocity_jump_event_chain)
        self.state = None
