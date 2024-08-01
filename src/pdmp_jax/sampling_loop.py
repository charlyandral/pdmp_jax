import jax
import jax.numpy as jnp
from jax.tree_util import Partial as jax_partial

from .namedtuples import PdmpState, PdmpOutput
from .upper_bound import next_event

def output_state(state: PdmpState) -> PdmpOutput:
    """
    Converts the given PdmpState object into a PdmpOutput object by selecting the relevant fields.

    Args:
        state (PdmpState): The PdmpState object to convert.

    Returns:
        PdmpOutput: The converted PdmpOutput object.
    """
    keys = PdmpOutput._fields
    values = state._asdict()
    return PdmpOutput(**{key: values[key] for key in keys})


def compare_pdmp_states(state1, state2):
    # Convert namedtuples to dictionaries
    dict1 = state1._asdict()
    dict2 = state2._asdict()

    # Find and return differences
    for key, value in dict1.items():
        if type(value) == type(state1.integrator):
            continue
        try:
            if value != dict2[key]:
                print(f"{key} is different, value: {value} != {dict2[key]}")
        except (ValueError, TypeError):
            try:
                if jnp.any(value != dict2[key]):
                    print(f"{key} is different, value: {value} != {dict2[key]}")
            except (ValueError, TypeError):
                print(f"{key} is different, error")


def move_before_horizon(state: PdmpState) -> PdmpState:
    accept = False
    state = state._replace(accept=accept)

    def cond(state):
        return jnp.logical_and(state.tp < state.horizon, jnp.logical_not(state.accept))

    state = jax.lax.while_loop(cond, inner_while, state)
    return state


def error_acceptance(state: PdmpState) -> PdmpState:
    horizon = state.horizon / 2
    upper_bound = state.upper_bound_func(state.x, state.v, horizon)
    key, subkey = jax.random.split(state.key)
    exp_rv = jax.random.exponential(subkey)
    tp, lambda_bar = next_event(upper_bound, exp_rv)
    horizon_new = jnp.where(state.adaptive, horizon, state.horizon)
    state = state._replace(
        horizon=horizon_new,
        tp=tp,
        exp_rv=exp_rv,
        key=key,
        lambda_bar=lambda_bar,
        error_bound=state.error_bound + 1,
        error_value_ar=state.error_value_ar.at[state.error_bound % 5].set(state.ar),
    )
    return state


def ok_acceptance(state: PdmpState) -> PdmpState:
    key, subkey = jax.random.split(state.key)
    accept = jax.random.bernoulli(subkey, state.ar)
    state = state._replace(lambda_t=state.lambda_t, accept=accept, key=key)
    state = jax.lax.cond(accept, if_accept, if_not_accept, state)
    cond = jnp.logical_and(state.tp > state.horizon, jnp.logical_not(state.accept))
    state = jax.lax.cond(cond, move_to_horizon2, lambda x: x, state)
    return state


def inner_while(state: PdmpState) -> PdmpState:
    lambda_t = state.rate(state.x, state.v, state.tp)
    ar = lambda_t / state.lambda_bar
    state = state._replace(lambda_t=lambda_t, ar=ar)
    state = jax.lax.cond(ar > 1.0, error_acceptance, ok_acceptance, state)
    return state


def move_to_horizon2(state: PdmpState) -> PdmpState:
    ts = state.ts + state.horizon
    xi, vi = state.integrator(state.x, state.v, state.horizon)
    state = state._replace(x=xi, v=vi, ts=ts, hitting_horizon=state.hitting_horizon + 1)
    return state


def if_accept(state: PdmpState) -> PdmpState:
    x, v = state.integrator(state.x, state.v, state.tp)
    key, subkey = jax.random.split(state.key)
    v = state.velocity_jump(x, v, subkey)  # type: ignore
    t = state.t + state.tp + state.ts
    indicator = True
    ts = 0.0
    tp = 0.0
    accept = True
    state = state._replace(
        x=x,
        v=v,
        t=t,
        indicator=indicator,
        ts=ts,
        tp=tp,
        accept=accept,
        key=key,
    )
    return state


def if_not_accept(state: PdmpState) -> PdmpState:
    key, subkey = jax.random.split(state.key)
    exp_rv = state.exp_rv + jax.random.exponential(subkey)
    tp, lambda_bar = next_event(state.upper_bound, exp_rv)
    horizon = jnp.where(state.adaptive, state.horizon / 1.04, state.horizon)
    state = state._replace(
        tp=tp,
        exp_rv=exp_rv,
        lambda_bar=lambda_bar,
        key=key,
        rejected=state.rejected + 1,
        horizon=horizon,
    )
    return state


def move_to_horizon(state: PdmpState) -> PdmpState:
    ts = state.ts + state.horizon
    xi, vi = state.integrator(state.x, state.v, state.horizon)
    horizon = jnp.where(state.adaptive, state.horizon * 1.01, state.horizon)
    state = state._replace(
        x=xi, v=vi, ts=ts, hitting_horizon=state.hitting_horizon + 1, horizon=horizon
    )
    return state


def one_step_while(state: PdmpState) -> PdmpState:
    upper_bound = state.upper_bound_func(state.x, state.v, state.horizon)
    key, subkey = jax.random.split(state.key)
    exp_rv = jax.random.exponential(subkey)
    tp, lambda_bar = next_event(upper_bound, exp_rv)
    cond = tp > state.horizon
    state = state._replace(
        tp=tp, exp_rv=exp_rv, lambda_bar=lambda_bar, key=key, upper_bound=upper_bound
    )
    state = jax.lax.cond(cond, move_to_horizon, move_before_horizon, state)
    return state


def one_step(state: PdmpState) -> PdmpState:
    def cond_fun(state):
        return jnp.logical_not(state.indicator)

    state = state._replace(
        error_bound=0, rejected=0, hitting_horizon=0, error_value_ar=jnp.zeros(5)
    )
    state = jax.lax.while_loop(cond_fun, one_step_while, state)
    state = state._replace(indicator=False)
    return state
