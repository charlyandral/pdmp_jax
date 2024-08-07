import jax
import jax.numpy as jnp
from jax.tree_util import Partial as jax_partial

import warnings

from .pdmp import PDMP


class NonExploSpeedupZigZag(PDMP):
    """
    NonExploSpeedupZigZag class for Speedup ZigZag sampler, in the non explosive setting.
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
        dim,
        grad_U,
        grid_size=10,
        tmax=1.0,
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

        # Define the integrator for the non-explosive setting
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
        
        # Modify the gradient of the potential used in the rate to include the change of speed
        self.speed = lambda x: jnp.sqrt(1.0 + x @ x)
        self.grad_speed = jax.grad(self.speed)
        self.true_grad_U = jax_partial(grad_U) # keep track of the true gradient
        self.grad_U = jax_partial(
            lambda x: self.speed(x) * self.true_grad_U(x) - self.grad_speed(x)
        )
        # Initialize the rate with the modified gradient
        self.rate, self.rate_vect, self.signed_rate, self.signed_rate_vect = (
            self._init_zz_rate()
        )
        # Define the velocity jump function
        def _velocity_jump_zz(x, v, key):
            lambda_t = jnp.maximum(0.0, self.grad_U(x) * v)
            proba = lambda_t / jnp.sum(lambda_t)
            m = jax.random.choice(key, jnp.arange(v.shape[0]), p=proba)
            v = v.at[m].mul(-1)
            return v

        self.velocity_jump = jax_partial(_velocity_jump_zz)

        self.state = None
