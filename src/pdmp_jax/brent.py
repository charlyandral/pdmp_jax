from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Float, Int


class BrentState(NamedTuple):
    func: Callable
    xatol: Float
    maxfun: Int
    sqrt_eps: Float
    golden_mean: Float
    a: Float
    b: Float
    fulc: Float
    nfc: Float
    xf: Float
    e: Float
    rat: Float
    fx: Float
    num: Int
    fnfc: Float
    ffulc: Float
    xm: Float
    tol1: Float
    tol2: Float
    golden: Int


def minimize_scalar_bounded_jax(func, bounds, xatol=1e-7, maxiter=500):
    """Jax implementation of minimize_scalar_bounded from scipy.optimize.

    Args:
        func (function): func must be able to be passed to a jax.jit function. For instance do jax.tree_util.Partial(func)
        bounds (tuple(float)): (lower_bound, upper_bound) for the minimization
        xatol (float, optional): . Defaults to 1e-5.
        maxiter (int, optional): maximum_number of iterations. Defaults to 500.
    """

    def cond_loop(val: BrentState):
        condi = jnp.bool_(jnp.abs(val.xf - val.xm) > (val.tol2 - 0.5 * (val.b - val.a)))
        return condi & (val.num < val.maxfun)

    def loop_while(state: BrentState) -> BrentState:
        # Start assuming golden-section step; inner_if_1 may switch to parabolic and set golden=0

        # Check for parabolic fit
        def inner_if_1(s: BrentState) -> BrentState:
            # Default to parabolic attempt (golden=0). Update e to previous rat.
            # Parabolic interpolation terms computed directly from state
            r_par = (s.xf - s.nfc) * (s.fx - s.ffulc)
            q_par = (s.xf - s.fulc) * (s.fx - s.fnfc)
            p_par = (s.xf - s.fulc) * q_par - (s.xf - s.nfc) * r_par
            q_par = 2.0 * (q_par - r_par)
            p_par = jnp.where(q_par > 0.0, -p_par, p_par)
            q_abs = jnp.abs(q_par)
            r_par = s.e
            e_new = s.rat

            def fun_true(s_in: BrentState) -> BrentState:
                # Accept parabola: keep golden=0, update rat to parabolic step (with safeguard near boundaries)
                rat_new = jnp.where(
                    ((s.xf - s.a) < s.tol2) | ((s.b - s.xf) < s.tol2),
                    s.tol1 * (jnp.sign(s.xm - s.xf) + ((s.xm - s.xf) == 0)),
                    (p_par + 0.0) / q_abs,
                )
                return s_in._replace(
                    e=e_new,
                    rat=rat_new,
                    golden=jnp.asarray(0),
                )

            def fun_false(s_in: BrentState) -> BrentState:
                # Reject parabola: indicate golden step will be used next
                return s_in._replace(
                    e=e_new,
                    golden=jnp.asarray(1),
                )

            cond_ok = (
                (jnp.abs(p_par) < jnp.abs(0.5 * q_abs * r_par))
                & (p_par > q_abs * (s.a - s.xf))
                & (p_par < q_abs * (s.b - s.xf))
            )
            return jax.lax.cond(cond_ok, fun_true, fun_false, s)

        state2 = state._replace(golden=jnp.asarray(1))

        state2 = jax.lax.cond(
            jnp.abs(state2.e) > state2.tol1, inner_if_1, lambda s: s, state2
        )

        # If parabola rejected (golden==1), set golden-section step
        def _golden_true(s: BrentState):
            step = jnp.where(s.xf >= s.xm, s.a - s.xf, s.b - s.xf)
            return s._replace(
                e=step,
                rat=s.golden_mean * step,
            )

        def _golden_false(s: BrentState):
            return s

        state2 = jax.lax.cond(state2.golden == 1, _golden_true, _golden_false, state2)

        # Trial point and function evaluation
        x = state2.xf + (jnp.sign(state2.rat) + (state2.rat == 0)) * jnp.maximum(
            jnp.abs(state2.rat), state2.tol1
        )
        fu = state2.func(x)
        num = state2.num + 1

        def inner_if_2T(s: BrentState) -> BrentState:
            return s._replace(
                a=jnp.where(x >= s.xf, s.xf, s.a),
                b=jnp.where(x >= s.xf, s.b, s.xf),
                fulc=s.nfc,
                nfc=s.xf,
                xf=x,
                fx=fu,
                num=num,
                fnfc=s.fx,
                ffulc=s.fnfc,
            )

        def inner_if_2F(s: BrentState) -> BrentState:
            cond1 = (fu <= s.fnfc) | (s.nfc == s.xf)
            cond2 = (
                (fu <= s.ffulc) | (s.fulc == s.xf) | (s.fulc == s.nfc)
            ) & jnp.logical_not(cond1)
            return s._replace(
                a=jnp.where(x < s.xf, x, s.a),
                b=jnp.where(x < s.xf, s.b, x),
                fulc=jnp.where(cond1, s.nfc, jnp.where(cond2, x, s.fulc)),
                nfc=jnp.where(cond1, x, s.nfc),
                num=num,
                fnfc=jnp.where(cond1, fu, s.fnfc),
                ffulc=jnp.where(cond1, s.fnfc, jnp.where(cond2, fu, s.ffulc)),
            )

        state3 = jax.lax.cond(fu <= state2.fx, inner_if_2T, inner_if_2F, state2)

        # Update mid and tolerances
        xm = 0.5 * (state3.a + state3.b)
        tol1 = state3.sqrt_eps * jnp.abs(state3.xf) + state3.xatol / 3.0
        tol2 = 2.0 * tol1

        return state3._replace(xm=xm, tol1=tol1, tol2=tol2)

    xatol = jnp.asarray(xatol)
    maxfun = jnp.asarray(maxiter)
    x1, x2 = bounds
    flag = 0
    sqrt_eps = jnp.sqrt(2.2e-16)
    golden_mean = 0.5 * (3.0 - jnp.sqrt(5.0))
    a, b = x1, x2
    a = jnp.asarray(a)
    b = jnp.asarray(b)
    fulc = a + golden_mean * (b - a)
    nfc, xf = fulc, fulc
    rat = e = jnp.asarray(0.0)
    x = xf
    fx = func(x)
    num = jnp.asarray(1)

    ffulc = fnfc = fx
    xm = 0.5 * (a + b)
    tol1 = sqrt_eps * jnp.abs(xf) + xatol / 3.0
    tol2 = 2.0 * tol1

    fval = fx
    golden = 0
    val = BrentState(
        func,
        xatol,
        maxfun,
        sqrt_eps,
        golden_mean,
        a,
        b,
        fulc,
        nfc,
        xf,
        e,
        rat,
        fx,
        num,
        fnfc,
        ffulc,
        xm,
        tol1,
        tol2,
        golden,
    )

    val = jax.lax.while_loop(cond_loop, loop_while, val)
    # Extract results from final state
    fval = val.fx
    flag = jnp.where(val.num >= val.maxfun, False, True)
    result = {
        "min": fval,
        "argmin": val.xf,
        "success": flag,
        "evals": val.num,
        "lower_bound": val.a,
        "upper_bound": val.b,
    }
    return result
