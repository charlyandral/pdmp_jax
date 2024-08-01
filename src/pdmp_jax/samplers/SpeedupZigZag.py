import jax
import jax.numpy as jnp
from jax.tree_util import Partial as jax_partial

import warnings

from .pdmp import PDMP


class NonExploSpeedupZigZag(PDMP):
    def __init__(
        self,
        dim,
        grad_U,
        grid_size=100,
        tmax=2.0,
        vectorized_bound=True,
        signed_bound=True,
        adaptive=True,
        **kwargs,
    ):

        self.dim = dim
        self.refresh_rate = 0.0
        self.grid_size = grid_size
        self.tmax = float(tmax)
        self.vectorized_bound = vectorized_bound

        if signed_bound and (not vectorized_bound):
            self.signed_bound = False
            warnings.warn(
                "Signed bound is not compatible with non-vectorized bound for ZigZag switching to unsigned bound"
            )
        else:
            self.signed_bound = signed_bound
        self.adaptive = adaptive

        def integrator_path_non_explo(x, v, t):
            d = x.shape[0]
            y = x - v[0] * x[0] * v
            c = v[0] * (y @ v)
            a = (1 + y @ y) / d - (c**2) / (d**2)
            Y_0 = x[0] + (c / d)
            b_t = (Y_0 + jnp.sqrt(Y_0**2 + a)) * jnp.exp(jnp.sqrt(d) * v[0] * t)
            X_1 = (b_t**2 - a) / (2 * b_t) - (c / d)
            return y + v[0] * X_1 * v, v

        self.integrator = jax_partial(integrator_path_non_explo)
        self.speed = lambda x: jnp.sqrt(1.0 + x @ x)
        self.grad_speed = jax.grad(self.speed)
        self.true_grad_U = jax_partial(grad_U)
        self.grad_U = jax_partial(
            lambda x: self.speed(x) * self.true_grad_U(x) - self.grad_speed(x)
        )

        self.rate, self.rate_vect, self.signed_rate, self.signed_rate_vect = (
            self._init_zz_rate()
        )

        def _velocity_jump_zz(x, v, key):
            lambda_t = jnp.maximum(0.0, self.grad_U(x) * v)
            proba = lambda_t / jnp.sum(lambda_t)
            m = jax.random.choice(key, jnp.arange(v.shape[0]), p=proba)
            v = v.at[m].mul(-1)
            return v

        self.velocity_jump = jax_partial(_velocity_jump_zz)

        self.state = None
