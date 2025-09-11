import jax
import jax.numpy as jnp
from jax.tree_util import Partial as jax_partial

from typing import  Callable
from jaxtyping import  Array


from .pdmp import PDMP


class ForwardEventChain(PDMP):
    """
    ForwardEventChain class for the Forward Event Chain sampler.
    Args:
        dim (int): The dimension of the space.
        grad_U (Callable[[Array], Array]): The gradient of the potential energy function.
        grid_size (int, optional): The number of grid points for discretizing the space. Defaults to 10.
        tmax (float, optional): The horizon for the grid. Defaults to 1.0. If 0, adaptive tmax is used.
        refresh_ortho (float, optional): The refresh rate for the orthogonal component, i.e. the probability to perform the rotation on the orthogonal space. Defaults to 0.1.
        signed_bound (bool, optional): Whether to use signed bound strategy. Defaults to True.
        adaptive (bool, optional): Whether to use adaptive tmax Defaults to True.
        **kwargs: Additional keyword arguments.
    Attributes:
        dim (int): The dimension of the space.
        refresh_rate (float): The refresh rate. Is always 0 in the Forward Event Chain.
        refresh_ortho (float): The refresh rate for the orthogonal component.
        grad_U (Callable[[Array], Array]): The gradient of the potential.
        grid_size (int): The number of grid points for discretizing the space.
        tmax (float): The tmax for the grid.
        adaptive (bool): Whether to use adaptive tmax.
        vectorized_bound (bool): Whether to use vectorized strategy.
        signed_bound (bool): Whether to use signed strategy.
        integrator (Callable[[Array, Array, float], Tuple[Array, Array]]): The integrator function.
        rate (Array): The rate of the process.
        rate_vect (Array): The vectorized rate. Not used in the Forward Event Chain.
        signed_rate (Array): The signed rate.
        signed_rate_vect (Array): The vectorized and signed rate. Not used in the Forward Event Chain.
        velocity_jump (Callable[[Array, Array, Any], Array]): The velocity jump function.
        state (Any): The state of the ZigZag sampler.
    """
    
    def __init__(
        self,
        dim,
        grad_U,
        grid_size=10,
        tmax=2.0,
        refresh_ortho=0.1,
        signed_bound=True,
        adaptive=True,
        **kwargs,
    ):  
        # Check if the dimension is greater than 2
        if dim <= 2:
            raise ValueError(
                "The dimension must be greater than 2 to use the ForwardEventChain"
            )
        self.dim = dim
        self.refresh_rate = 0.0  # no refresh rate is used in the forward event chain
        self.refresh_ortho = refresh_ortho
        self.grad_U = jax_partial(grad_U)
        self.grid_size = grid_size
        # adaptive tmax if tmax is 0
        if tmax == 0:
            self.tmax = 1.0
            self.adaptive = True
        else:
            self.tmax = float(tmax)
            self.adaptive = adaptive
        self.vectorized_bound = False # vectorized strategy is not used in the forward event chain
        self.signed_bound = signed_bound

        # define the integrator function
        self.integrator = jax_partial(lambda x, v, t: (x + (v * t), v))

        #initialize the rate
        self.rate, self.rate_vect, self.signed_rate, self.signed_rate_vect = (
            self._init_bps_rate()
        )

        # define the velocity jump function
        def _velocity_jump_event_chain(x, v, key):
            subkey1, subkey2, subkey3, subkey4 = jax.random.split(key, 4)
            dim = x.shape[0]
            u = jax.random.uniform(subkey1)
            rho = -((1 - u ** (2.0 / (dim - 1.0))) ** (0.5))
            grad_U_x = self.grad_U(x)
            grad_U_x = grad_U_x / jnp.linalg.norm(grad_U_x)
            v_par = (v @ grad_U_x) * grad_U_x
            v_ortho = v - v_par
            v_ortho = jax.lax.cond(
            jnp.linalg.norm(v_ortho) < 1e-10,
            lambda _: regenerate_orthogonal_vector(subkey4),
            lambda _: v_ortho,
            operand=None
            )

            def regenerate_orthogonal_vector(key):
                v_ortho_new = jax.random.normal(key, shape=(dim,))
                v_ortho_new -= (v_ortho_new @ grad_U_x) * grad_U_x
                return v_ortho_new

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
