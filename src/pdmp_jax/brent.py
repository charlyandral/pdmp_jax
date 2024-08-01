
import jax
import jax.numpy as jnp

def minimize_scalar_bounded_jax(func, bounds, xatol=1e-7, maxiter=500):
    """Jax implementation of minimize_scalar_bounded from scipy.optimize.

    Args:
        func (function): func must be able to be passed to a jax.jit function. For instance do jax.tree_util.Partial(func)
        bounds (tuple(float)): (lower_bound, upper_bound) for the minimization
        xatol (float, optional): . Defaults to 1e-5.
        maxiter (int, optional): maximum_number of iterations. Defaults to 500.
    """

    def cond_loop(val):
        (
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
        ) = val
        condi = jnp.bool_(jnp.abs(xf - xm) > (tol2 - 0.5 * (b - a)))
        return condi & (num < maxfun)

    def loop_while(val):
        (
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
        ) = val
        golden = jnp.asarray(1)

        # Check for parabolic fit
        def inner_if_1(val):
            (
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
            ) = val
            golden = 0
            r = (xf - nfc) * (fx - ffulc)
            q = (xf - fulc) * (fx - fnfc)
            p = (xf - fulc) * q - (xf - nfc) * r
            q = 2.0 * (q - r)
            p = jnp.where(q > 0.0, -p, p)
            q = jnp.abs(q)
            r = e
            e = rat
            x = xf
            # Check for acceptability of parabola
            fun_true = (
                lambda a, b, x, fulc, nfc, xf, e, rat, fx, num, fnfc, ffulc, golden: (
                    a,
                    b,
                    xf + (p + 0.0) / q,
                    fulc,
                    nfc,
                    xf,
                    e,
                    jnp.where(
                        ((x - a) < tol2) | ((b - x) < tol2),
                        tol1 * (jnp.sign(xm - xf) + ((xm - xf) == 0)),
                        (p + 0.0) / q,
                    ),
                    fx,
                    num,
                    fnfc,
                    ffulc,
                    golden,
                )
            )

            fun_false = (
                lambda a, b, x, fulc, nfc, xf, e, rat, fx, num, fnfc, ffulc, golden: (
                    a,
                    b,
                    xf,
                    fulc,
                    nfc,
                    xf,
                    e,
                    rat,
                    fx,
                    num,
                    fnfc,
                    ffulc,
                    jnp.asarray(1),
                )
            )

            a, b, x, fulc, nfc, xf, e, rat, fx, num, fnfc, ffulc, golden = jax.lax.cond(
                (jnp.abs(p) < jnp.abs(0.5 * q * r))
                & (p > q * (a - xf))
                & (p < q * (b - xf)),
                fun_true,
                fun_false,
                a,
                b,
                x,
                fulc,
                nfc,
                xf,
                e,
                rat,
                fx,
                num,
                fnfc,
                ffulc,
                golden,
            )

            return (
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

        val = (
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

        (
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
        ) = jax.lax.cond(jnp.abs(e) > tol1, inner_if_1, lambda _val: _val, val)

        e, rat = jax.lax.cond(
            golden == 1,
            lambda e, rat: (
                jnp.where(xf >= xm, a - xf, b - xf),
                golden_mean * jnp.where(xf >= xm, a - xf, b - xf),
            ),
            lambda e, rat: (e, rat),
            e,
            rat,
        )
        x = xf + (jnp.sign(rat) + (rat == 0)) * jnp.maximum(jnp.abs(rat), tol1)
        fu = func(x)
        num += 1

        def inner_if_2T(a, b, x, xf, fx, fulc, ffulc, nfc, fnfc):
            a = jnp.where(x >= xf, xf, a)
            b = jnp.where(x >= xf, b, xf)
            fulc, ffulc = nfc, fnfc
            nfc, fnfc = xf, fx
            xf, fx = x, fu
            return a, b, x, xf, fx, fulc, ffulc, nfc, fnfc

        def inner_if_2F(a, b, x, xf, fx, fulc, ffulc, nfc, fnfc):
            a = jnp.where(x < xf, x, a)
            b = jnp.where(x < xf, b, x)
            cond1 = (fu <= fnfc) | (nfc == xf)
            cond2 = ((fu <= ffulc) | (fulc == xf) | (fulc == nfc)) & jnp.logical_not(
                cond1
            )
            fulc = jnp.where(cond2, x, fulc)
            ffulc = jnp.where(cond2, fu, ffulc)

            fulc = jnp.where(cond1, nfc, fulc)
            ffulc = jnp.where(cond1, fnfc, ffulc)
            nfc = jnp.where(cond1, x, nfc)    
            fnfc = jnp.where(cond1, fu, fnfc)
            return a, b, x, xf, fx, fulc, ffulc, nfc, fnfc

        a, b, x, xf, fx, fulc, ffulc, nfc, fnfc = jax.lax.cond(
            fu <= fx, inner_if_2T, inner_if_2F, a, b, x, xf, fx, fulc, ffulc, nfc, fnfc
        )

        xm = 0.5 * (a + b)
        tol1 = sqrt_eps * jnp.abs(xf) + xatol / 3.0
        tol2 = 2.0 * tol1

        return (
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

    fu = jnp.inf

    ffulc = fnfc = fx
    xm = 0.5 * (a + b)
    tol1 = sqrt_eps * jnp.abs(xf) + xatol / 3.0
    tol2 = 2.0 * tol1

    fval = fx
    golden = 0
    val = (
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
    (
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
    ) = val
    fval = fx
    flag = jnp.where(num >= maxfun, False, True)
    result = {
        "min": fval,
        "argmin": xf,
        "success": flag,
        "evals": num,
        "lower_bound": a,
        "upper_bound": b,
    }
    return result
