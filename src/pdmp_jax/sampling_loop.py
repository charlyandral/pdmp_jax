from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from .namedtuples import PdmpOutput
from .upper_bound import next_event

if TYPE_CHECKING:
    from jaxtyping import Array, Bool

    from .namedtuples import PdmpState


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


def compare_pdmp_states(state1: PdmpState, state2: PdmpState) -> None:
    # Convert namedtuples to dictionaries
    dict1 = state1._asdict()
    dict2 = state2._asdict()

    # Find and return differences
    for key, value in dict1.items():
        if type(value) is type(state1.integrator):
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
    """Run the inner thinning loop using the bound's existing grid."""
    state = state._replace(accept=jnp.array(False))

    def cond(state: PdmpState) -> Bool[Array, ""]:
        # tp=inf exits after either a window end or a request to rebuild.
        # The latter keeps x, v and both clocks unchanged.
        return jnp.logical_and(state.tp <= state.horizon, jnp.logical_not(state.accept))

    return jax.lax.while_loop(cond, inner_while, state)


def record_bound_error(state: PdmpState) -> PdmpState:
    """Record a detected violation without deciding how to handle it."""
    return state._replace(
        error_bound=state.error_bound + 1,
        error_value_ar=state.error_value_ar.at[state.error_bound % 5].set(state.ar),
    )


def request_bound_rebuild(state: PdmpState) -> PdmpState:
    """Halve the horizon and exit the inner loop without advancing the flow.

    one_step_while will construct the replacement bound and draw a fresh
    candidate on the next outer iteration. No horizon hit is recorded here.
    """
    return state._replace(
        horizon=state.horizon / 2,
        tp=jnp.full_like(state.tp, jnp.inf),
        accept=jnp.array(False),
    )


def ok_acceptance(state: PdmpState) -> PdmpState:
    key, subkey = jax.random.split(state.key)
    # Keep the raw ratio in state.ar for diagnostics, including values above 1.
    accept = jax.random.bernoulli(subkey, jnp.minimum(state.ar, 1.0))
    state = state._replace(lambda_t=state.lambda_t, accept=accept, key=key)
    state = jax.lax.cond(accept, if_accept, if_not_accept, state)
    return state


def inner_while(state: PdmpState) -> PdmpState:
    """Process one candidate inside the current horizon."""
    lambda_t = state.rate(state.x, state.v, state.tp)
    ar = lambda_t / state.lambda_bar
    state = state._replace(lambda_t=lambda_t, ar=ar)
    invalid = ar > 1.0
    state = jax.lax.cond(invalid, record_bound_error, lambda s: s, state)
    # Normal candidates and fixed-horizon violations share the same acceptance
    # path; ok_acceptance clips the probability but preserves the raw ratio.
    return jax.lax.cond(
        jnp.logical_and(invalid, state.adaptive),
        request_bound_rebuild,
        ok_acceptance,
        state,
    )


def if_accept(state: PdmpState) -> PdmpState:
    x, v = state.integrator(state.x, state.v, state.tp)
    key, subkey = jax.random.split(state.key)
    v = state.velocity_jump(x, v, subkey)  # type: ignore
    state = state._replace(
        x=x,
        v=v,
        t=state.t + state.ts + state.tp,
        ts=jnp.zeros_like(state.ts),
        tp=jnp.zeros_like(state.tp),
        indicator=jnp.array(True),
        accept=jnp.array(True),
        key=key,
    )
    return state


def if_not_accept(state: PdmpState) -> PdmpState:
    """Shrink immediately, without restarting before the rejected candidate."""
    assert state.upper_bound is not None
    rejected_time = state.tp
    key, subkey = jax.random.split(state.key)
    exp_rv = state.exp_rv + jax.random.exponential(subkey)
    tp, lambda_bar = next_event(state.upper_bound, exp_rv)
    horizon = jnp.where(
        state.adaptive, state.horizon / state.alpha_minus, state.horizon
    )
    state = state._replace(
        tp=tp,
        exp_rv=exp_rv,
        lambda_bar=lambda_bar,
        key=key,
        rejected=state.rejected + 1,
        horizon=horizon,
    )
    # Reuse the bound on the shortened interval. If shortening
    # passes the rejected candidate, that candidate is the earliest safe restart.
    reached = jnp.logical_or(rejected_time >= horizon, tp > horizon)
    return jax.lax.cond(
        reached,
        lambda s: move_to_horizon(s, jnp.maximum(rejected_time, horizon)),
        lambda s: s,
        state,
    )


def move_to_horizon(
    state: PdmpState, duration: float | Array | None = None
) -> PdmpState:
    if duration is None:
        duration = state.horizon
    x, v = state.integrator(state.x, state.v, duration)
    horizon = jnp.where(state.adaptive, state.horizon * state.alpha_plus, state.horizon)
    state = state._replace(
        x=x,
        v=v,
        ts=state.ts + duration,
        tp=jnp.full_like(state.tp, jnp.inf),
        hitting_horizon=state.hitting_horizon + 1,
        horizon=horizon,
    )
    return state


def one_step_while(state: PdmpState) -> PdmpState:
    """Construct a bound, then reuse it until acceptance or a window exit."""
    upper_bound = state.upper_bound_func(state.x, state.v, state.horizon)
    key, subkey = jax.random.split(state.key)
    exp_rv = jax.random.exponential(subkey)
    tp, lambda_bar = next_event(upper_bound, exp_rv)
    cond = tp > upper_bound.grid[-1]
    state = state._replace(
        tp=tp,
        exp_rv=exp_rv,
        lambda_bar=lambda_bar,
        key=key,
        upper_bound=upper_bound,
        bound_evals=state.bound_evals + upper_bound.evals,
        accept=jnp.array(False),
    )
    state = jax.lax.cond(cond, move_to_horizon, move_before_horizon, state)
    return state


def one_step(state: PdmpState) -> PdmpState:
    """Outer loop over bound constructions until the next accepted event."""
    def cond_fun(state):
        return jnp.logical_not(state.indicator)

    state = state._replace(
        error_bound=jnp.array(0),
        rejected=jnp.array(0),
        hitting_horizon=jnp.array(0),
        bound_evals=jnp.array(0),
        error_value_ar=jnp.zeros(5),
    )
    state = jax.lax.while_loop(cond_fun, one_step_while, state)
    state = state._replace(indicator=jnp.array(False))
    return state
