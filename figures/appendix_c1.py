# %%
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

os.environ["XLA_FLAGS"] = "--xla_cpu_use_thunk_runtime=false"

from itertools import product
from time import time

import jax
import jax.numpy as jnp
import numpy as np
from joblib import Parallel, delayed
from tqdm.notebook import tqdm

import pdmp_jax as pdmp

if TYPE_CHECKING:
    from jaxtyping import Array

    from pdmp_jax.namedtuples import PdmpOutput

print(jax.__version__)


dim = 2


sd = 0.03
inv_var_2 = 1 / sd**2

weights = jnp.array([1, 1 / (sd**dim)])


def U(x: Array) -> Array:
    x_shift = x - 1.0
    return -jax.nn.logsumexp(
        -jnp.array([x.T @ x / 2, x_shift.T @ (inv_var_2 * x_shift) / 2]), b=weights
    )


@jax.jit
def true_mean(out: PdmpOutput) -> Array:
    mean = (out.x[1:] + out.x[:-1]) / 2 * (out.t[1:] - out.t[:-1])[:, None]
    return jnp.sum(mean, axis=0) / (out.t[-1] - out.t[0])


# %%
grad_U = jax.grad(U)


def loop(grid_size, seed, tmax) -> dict[str, Any]:
    sampler = pdmp.BouncyParticle(dim, grad_U, grid_size, tmax, adaptive=False)
    seed1, seed2 = jax.random.split(jax.random.PRNGKey(seed))
    xinit = jax.random.normal(seed1, shape=(2,))
    vinit = jax.random.normal(seed2, shape=(2,))
    vinit = vinit / jnp.linalg.norm(vinit)
    begin = time()
    out = sampler.sample_skeleton(1000000, xinit, vinit, seed, verbose=False)
    out.x.block_until_ready()
    end = time()
    dico = out._asdict()
    dico["grid_size"] = grid_size
    dico["tmax"] = tmax
    dico["time"] = end - begin
    dico["mean_error"] = true_mean(out) - jnp.array([0.5, 0.5])
    # drop x, v and t from dico
    dico.pop("x")
    dico.pop("v")
    dico.pop("t")
    return dico


grid_sizes = [0, 5, 10, 20, 50, 100]
# grid_sizes = [5, 10, 20, 50, 100]
n_seed = 10
seeds = list(range(n_seed))
tmaxs = [0.0, 0.01, 0.1, 1.0]
# tmaxs = [0.0]
# vects_signeds = [(False,False),(True,False),(True,True)]
# vects_signeds = [(True,True)]
iterable = list(product(grid_sizes, seeds, tmaxs))


np.random.shuffle(iterable)  # type: ignore
print(len(iterable))
results_fec: list[dict] = Parallel(n_jobs=8, backend="loky")(
    delayed(loop)(*args) for args in tqdm(iterable)
)  # type: ignore

# %%
