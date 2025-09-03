import jax
import jax.numpy as jnp
from jax.tree_util import Partial as jax_partial

from pdmp_jax.brent import minimize_scalar_bounded_jax


def test_quadratic():
    # f(x) = (x - 2)^2 has minimum at x=2
    f = jax_partial(lambda x: (x - 2.0) ** 2)
    res = minimize_scalar_bounded_jax(f, bounds=(0.0, 5.0), xatol=1e-6, maxiter=200)
    # Basic assertions
    assert bool(res["success"]) is True, f"Expected success, got {res}"
    # JAX may use 32-bit; be tolerant
    assert jnp.abs(res["argmin"] - 2.0) < 1e-3, f"argmin off: {res['argmin']}"
    assert res["min"] >= 0.0


def test_quadratic_jitted():
    f = jax_partial(lambda x: (x - 2.0) ** 2)
    minimize_scalar_bounded_jax_jit = jax.jit(minimize_scalar_bounded_jax)
    res = minimize_scalar_bounded_jax_jit(f, bounds=(0.0, 5.0), xatol=1e-6, maxiter=200)
    assert bool(res["success"]) is True
