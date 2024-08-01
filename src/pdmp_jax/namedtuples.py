from typing import NamedTuple, Callable, Tuple
from jaxtyping import Float, Int, PRNGKeyArray, Bool, Array
import jax.numpy as jnp


class BoundBox(NamedTuple):
    """
    Represents the output of the upper bound function. Is a NamedTuple with the following fields.

    Attributes:
        grid (Float[Array, "n_grid"]): The grid values.
        box_max (Float[Array, "n_grid - 1"]): The maximum values on each segment of the grid.
        cum_sum (Float[Array, "n_grid - 1"]): The cumulative sum of box_max.
        step_size (Float[Array, ""]): The step size of the grid.
    """
    
    grid : Float[Array, "n_grid"]
    box_max : Float[Array, "n_grid - 1"]
    cum_sum : Float[Array, "n_grid - 1"]
    step_size: Float[Array, ""]



class PdmpState(NamedTuple):
    """
    Represents the state of the PDMP sampler. Is a NamedTuple with the following fields.

    Attributes:
        x (Array[float, "dim"]): position
        v (Array[float, "dim"]): velocity
        t (Array[float, ""]): time
        horizon (Array[float, ""]): horizon
        key (PRNGKeyArray): random key
        integrator (Callable[[Array, Array, Array], Tuple[Array, Array]]): integrator function
        grad_U (Callable[[Array], Array]): gradient of the potential function
        rate (Callable[[Array, Array, Array], Array]): rate function
        velocity_jump (Callable[[Array, Array, PRNGKeyArray], Array]): velocity jump function
        upper_bound_func (Callable[[Array, Array, Array], aux.BoundBox]): upper bound function
        accept (Array[bool, ""]): accept indicator for the thinning
        upper_bound (aux.BoundBox | None): upper bound box
        indicator (Array[bool, ""]): indicator for jumping
        tp (Array[float, ""]): time to the next event
        ts (Array[float, ""]): time spent
        exp_rv (Array[float, ""]): exponential random variable for the Poisson process
        lambda_bar (Array[float, ""]): upper bound for the Poisson process
        lambda_t (Array[float, ""]): rate at the current time
        ar (Array[float, ""]): acceptance rate for the thinning
        error_bound (Array[int, ""]): count of the number of errors in the upper bound
        rejected (Array[int, ""]): count of the number of rejections in the thinning
        hitting_horizon (Array[int, ""]): count of the number of hits of the horizon
    """

    x: Float[Array, "dim"]
    v: Float[Array, "dim"]
    t: Float[Array, ""]
    horizon: Float[Array, ""]
    key: PRNGKeyArray
    integrator: Callable[[Array, Array, Array], Tuple[Array, Array]]
    grad_U: Callable[[Array], Array]
    rate: Callable[[Array, Array, Array], Array]
    velocity_jump: Callable[[Array, Array, PRNGKeyArray], Array]
    upper_bound_func: Callable[[Array, Array, Array], BoundBox]
    accept: Bool[Array, ""] = jnp.array(False)
    upper_bound: BoundBox | None = None
    indicator: Bool[Array, ""] = jnp.array(False)
    tp: Float[Array, ""] = jnp.array(0.0)
    ts: Float[Array, ""] = jnp.array(0.0)
    exp_rv: Float[Array, ""] = jnp.array(0.0)
    lambda_bar: Float[Array, ""] = jnp.array(0.0)
    lambda_t: Float[Array, ""] = jnp.array(0.0)
    ar: Float[Array, ""] = jnp.array(0.0)
    error_bound: Int[Array, ""] = jnp.array(0)
    error_value_ar: Float[Array, ""] = jnp.zeros(5)
    rejected: Int[Array, ""] = jnp.array(0)
    hitting_horizon: Int[Array, ""] = jnp.array(0)
    adaptive: Bool[Array, ""] = jnp.array(False)


class PdmpOutput(NamedTuple):
    """
    Represents the output of a PDMP (Piecewise Deterministic Markov Process) simulation. Is a NamedTuple with the following fields.

    Attributes:
        x (Float[Array, "dim"]): The state trajectory.
        v (Float[Array, "dim"]): The velocity trajectory.
        t (Float[Array, ""]): The time points at which the state and velocity are recorded.
        error_bound (Int[Array, ""]): The error bound at each time point.
        rejected (Int[Array, ""]): The indicator of whether a jump was rejected at each time point.
        hitting_horizon (Int[Array, ""]): The indicator of whether the process hit the horizon at each time point.
    """

    x: Float[Array, "dim"]
    v: Float[Array, "dim"]
    t: Float[Array, ""]
    error_bound: Int[Array, ""]
    error_value_ar: Float[Array, ""]
    rejected: Int[Array, ""]
    hitting_horizon: Int[Array, ""]
    ar: Float[Array, ""]
    horizon: Float[Array, ""]
