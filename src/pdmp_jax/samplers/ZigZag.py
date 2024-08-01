import jax
import jax.numpy as jnp
from jax.tree_util import Partial as jax_partial

from typing import  Callable
from jaxtyping import  Bool, Array

import warnings

from .pdmp import PDMP


class ZigZag(PDMP):
    def __init__(
        self,
        dim: int,
        grad_U: Callable[[Array], Array],
        grid_size: int = 100,
        tmax: float = 2.0,
        vectorized_bound: Bool = True,
        signed_bound: Bool = True,
        adaptive: Bool = True,
        **kwargs,
    ):
        self.dim = dim
        self.refresh_rate = 0.0
        self.grad_U = jax_partial(grad_U)
        self.grid_size = grid_size
        if tmax == 0:
            self.tmax = 1.0
            self.adaptive = True
        else:
            self.tmax = float(tmax)
            self.adaptive = adaptive

        self.vectorized_bound = vectorized_bound

        if signed_bound and (not vectorized_bound):
            self.signed_bound = False

            warnings.warn(
                "Signed bound is not compatible with non-vectorized bound for ZigZag switching to unsigned bound"
            )
        else:
            self.signed_bound = signed_bound

        self.integrator = jax_partial(lambda x, v, t: (x + (v * t), v))

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
