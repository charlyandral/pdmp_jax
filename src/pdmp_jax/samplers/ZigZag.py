import jax
import jax.numpy as jnp
from jax.tree_util import Partial as jax_partial

from typing import  Callable
from jaxtyping import  Bool, Array

import warnings

from .pdmp import PDMP


class ZigZag(PDMP):
    """
    ZigZag class for the ZigZag sampler.
    Args:
        dim (int): The dimension of the space.
        grad_U (Callable[[Array], Array]): The gradient of the potential energy function.
        grid_size (int, optional): The number of grid points for discretizing the space. Defaults to 10.
        tmax (float, optional): The horizon for the grid. Defaults to 1.0. If 0, adaptive tmax is used.
        vectorized_bound (bool, optional): Whether to use vectorized strategy for the bound. Defaults to True.
        signed_bound (bool, optional): Whether to use signed bound strategy. Defaults to True.
        adaptive (bool, optional): Whether to use adaptive tmax Defaults to True.
        **kwargs: Additional keyword arguments.
    Attributes:
        dim (int): The dimension of the space.
        refresh_rate (float): The refresh rate.
        grad_U (Callable[[Array], Array]): The gradient of the potential.
        grid_size (int): The number of grid points for discretizing the space.
        tmax (float): The tmax for the grid.
        adaptive (bool): Whether to use adaptive tmax.
        vectorized_bound (bool): Whether to use vectorized strategy.
        signed_bound (bool): Whether to use signed strategy.
        integrator (Callable[[Array, Array, float], Tuple[Array, Array]]): The integrator function.
        rate (Array): The rate of the process.
        rate_vect (Array): The vectorized rate.
        signed_rate (Array): The signed rate.
        signed_rate_vect (Array): The vectorized and signed rate.
        velocity_jump (Callable[[Array, Array, Any], Array]): The velocity jump function.
        state (Any): The state of the ZigZag sampler.
    """
    
    def __init__(
        self,
        dim: int,
        grad_U: Callable[[Array], Array],
        grid_size: int = 10,
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
        if tmax == 0: # adaptive tmax if tmax is 0
            self.tmax = 1.0
            self.adaptive = True
        else:
            self.tmax = float(tmax)
            self.adaptive = adaptive

        self.vectorized_bound = vectorized_bound

        # Specific warning for ZigZag
        if signed_bound and (not vectorized_bound):
            self.signed_bound = False

            warnings.warn(
                "Signed bound is not compatible with non-vectorized bound for ZigZag switching to unsigned bound"
            )
        else:
            self.signed_bound = signed_bound

        # definition of the integrator
        self.integrator = jax_partial(lambda x, v, t: (x + (v * t), v))

        # initialization of the rate
        self.rate, self.rate_vect, self.signed_rate, self.signed_rate_vect = (
            self._init_zz_rate()
        )

        # initialization of the velocity jump
        def _velocity_jump_zz(x, v, key):
            lambda_t = jnp.maximum(0.0, self.grad_U(x) * v)
            proba = lambda_t / jnp.sum(lambda_t)
            m = jax.random.choice(key, jnp.arange(v.shape[0]), p=proba)
            v = v.at[m].mul(-1)
            return v

        self.velocity_jump = jax_partial(_velocity_jump_zz)

        self.state = None
