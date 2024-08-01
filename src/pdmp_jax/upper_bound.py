import jax
import jax.numpy as jnp
from jax.tree_util import Partial as jax_partial


from .brent import minimize_scalar_bounded_jax
from .namedtuples import BoundBox


def upper_bound_constant(func, a, b, n_grid=100, refresh_rate=0.):
    """
    Computes the constant upper bound using the Brent's algorithm.

    Parameters:
    - func: The function for which the upper bound constant is computed.
    - a: The lower bound of the interval.
    - b: The upper bound of the interval.
    - n_grid: The number of grid points used for computation (default: 100).
    - refresh_rate: The refresh rate for the upper bound constant (default: 0).

    Returns:
    - BoundBox: An object containing the upper bound constant information.

    """
    func_min = jax_partial(lambda t: -func(t))
    bound = minimize_scalar_bounded_jax(func_min, bounds=(a, b))
    t = jnp.linspace(a, b, 2)
    box_max = jnp.atleast_1d(-bound["min"])
    box_max += refresh_rate
    cum_sum = jnp.zeros(2)
    cum_sum = cum_sum.at[1:].set(box_max * (b - a))

    return BoundBox(t, box_max, cum_sum, b - a)




def upper_bound_grid(func,a,b,n_grid = 100,refresh_rate = 0.):
    """Compute the upper bound using a grid 

    Args:
        func: the function for which the upper bound is computed
        a (float) : the lower bound of the interval
        b (float): the upper bound of the interval
        n_grid (int, optional): size of the grid for the upperbound of func. Defaults to 100.
        refresh_rate (float, optional): refresh rate for the upper bound. Defaults to 0.
    Returns:
        - BoundBox: An object containing the upper bound constant information.
    """
    t = jnp.linspace(a,b,n_grid)
    step_size = t[1]-t[0]
    func_vmap = jax.vmap(jax.value_and_grad(func))
    values,grads = func_vmap(t)
    intersection_pos = (values[:-1] - values[1:] + grads[1:]*step_size) / (grads[1:] - grads[:-1])
    intersection_pos = jnp.nan_to_num(intersection_pos,nan = 0.)
    intersection_pos = jnp.clip(intersection_pos,0.,step_size)
    intersection = values[:-1] + grads[:-1]*intersection_pos
    box_max = jnp.maximum(values[:-1],values[1:])
    box_max = jnp.maximum(box_max,intersection)
    box_max = jnp.maximum(box_max,0.)
    box_max += refresh_rate
    cum_sum = jnp.zeros(n_grid)
    cum_sum = cum_sum.at[1:].set(jnp.cumsum(box_max)*step_size)
    return BoundBox(t,box_max,cum_sum,step_size)


def upper_bound_grid_vect(func,a,b,n_grid = 100):
    """Compute the upper bound using a grid with the vectorized strategy

    Args:
        func: the function for which the upper bound is computed
        a (float) : the lower bound of the interval
        b (float): the upper bound of the interval
        n_grid (int, optional): size of the grid for the upperbound of func. Defaults to 100.
        refresh_rate (float, optional): refresh rate for the upper bound. Defaults to 0.
    Returns:
        - BoundBox: An object containing the upper bound constant information.
    """
    t = jnp.linspace(a,b,n_grid)
    step_size = t[1]-t[0]
    values,grads = jax.jvp(jax.vmap(func),(t,), (jnp.ones(t.size),))
    intersection_pos = (values[:-1] - values[1:] + grads[1:]*step_size) / (grads[1:] - grads[:-1])
    intersection_pos = jnp.nan_to_num(intersection_pos,nan = 0.)
    intersection_pos = jnp.clip(intersection_pos,0.,step_size)
    intersection = values[:-1] + grads[:-1]*intersection_pos
    box_max = jnp.maximum(values[:-1],values[1:])
    box_max = jnp.maximum(box_max,intersection)
    box_max = jnp.maximum(box_max,0.)
    cum_sum = jnp.zeros((n_grid,values.shape[1]))
    cum_sum = cum_sum.at[1:].set(jnp.cumsum(box_max,axis=0)*step_size)
    
    cum_sum = jnp.sum(cum_sum,axis = 1)
    box_max = jnp.sum(box_max,axis = 1)
    return BoundBox(t,box_max,cum_sum,step_size)



def next_event(boundbox,exp_rv):
    """
    Calculate the next event time based on the given exponential random variable and the upper bound.

    Args:
        boundbox (BoundBox): The boundbox object containing the cumulative sum and grid values.
        exp_rv (float): The exponential random variable.

    Returns:
        tuple: A tuple containing the next event time (t_prop) and the corresponding upper bound value.
    """
    
    index = jnp.searchsorted(boundbox.cum_sum,exp_rv) 
    # if the index is the last element, meaning that exp_rv > cum_sum[-1], it returns infinity
    t_prop = boundbox.grid[index-1] + (exp_rv - boundbox.cum_sum[index-1]) / (boundbox.cum_sum[index] - boundbox.cum_sum[index-1]) * boundbox.step_size
    return t_prop, boundbox.box_max[index-1]
