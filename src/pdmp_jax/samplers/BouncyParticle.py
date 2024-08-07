import jax
import jax.numpy as jnp
from jax.tree_util import Partial as jax_partial

from typing import  Callable
from jaxtyping import  Array


from .pdmp import PDMP


class BouncyParticle(PDMP):
    """
    BouncyParticle class for the Bouncy Particle sampler.
    Args:
        dim (int): The dimension of the space.
        grad_U (Callable[[Array], Array]): The gradient of the potential energy function.
        grid_size (int, optional): The number of grid points for discretizing the space. Defaults to 10.
        tmax (float, optional): The horizon for the grid. Defaults to 1.0. If 0, adaptive tmax is used.
        refresh_rate (float, optional): The refresh rate of the process. Defaults to 0.1.
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
        vectorized_bound (bool): Whether to use vectorized strategy. Not used in the Bouncy Particle.
        signed_bound (bool): Whether to use signed strategy.
        integrator (Callable[[Array, Array, float], Tuple[Array, Array]]): The integrator function.
        rate (Array): The rate of the process
        rate_vect (Array): The vectorized rate. Not used in the Bouncy Particle.
        signed_rate (Array): The signed rate.
        signed_rate_vect (Array): The vectorized and signed rate. Not used in the Bouncy Particle.
        velocity_jump (Callable[[Array, Array, Any], Array]): The velocity jump function.
        state (Any): The state of the ZigZag sampler.
    """
    def __init__(
        self,
        dim: int,
        grad_U: Callable[[Array], Array],
        grid_size: int = 10,
        tmax: float = 1.0,
        refresh_rate: float = 0.1,
        signed_bound: bool = True,
        adaptive: bool = True,
        **kwargs,
    ):
        
        self.dim = dim
        self.refresh_rate = refresh_rate
        self.grad_U = jax_partial(grad_U)
        self.grid_size = grid_size
        
        # adaptive tmax if tmax is 0
        if tmax == 0:
            self.tmax = 1.0
            self.adaptive = True
        else:
            self.tmax = float(tmax)
            self.adaptive = adaptive
        self.vectorized_bound = False # not used in the Bouncy Particle
        self.signed_bound = signed_bound

        # definition of the integrator
        self.integrator = jax_partial(lambda x, v, t: (x + (v * t), v))

        # initialization of the rate
        self.rate, self.rate_vect, self.signed_rate, self.signed_rate_vect = (
            self._init_bps_rate()
        )

        # definition of the velocity jump function
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

