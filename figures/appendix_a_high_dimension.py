# %%
from __future__ import annotations

import os
from typing import TYPE_CHECKING

from pdmp_jax.utils import (
    alpha_minus_plus_from_ratio_and_magnitude,
)

os.environ["XLA_FLAGS"] = "--xla_cpu_use_thunk_runtime=false"

from itertools import product
from time import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from tqdm.notebook import tqdm

import pdmp_jax as pdmp

if TYPE_CHECKING:
    from jaxtyping import Array

dim = 3


def U(x: Array) -> Array:
    # return x.T @ x / 2 #uncomment this for gaussian
    mean_x2 = x[0] ** 2 - 1
    return -(-(x[0] ** 2) + -((x[1] - mean_x2) ** 2) - jnp.sum((x[2:]) ** 2)) / 2


grad_U = jax.grad(U)


# %%
def loop(
    grid_size: int,
    seed: int,
    tmax: float,
    ratio_alphas: float,
    magnitude_alphas: float,
):
    alpha_minus, alpha_plus = alpha_minus_plus_from_ratio_and_magnitude(
        ratio_alphas, magnitude_alphas
    )
    sampler = pdmp.BouncyParticle(
        dim,
        grad_U,
        grid_size,
        tmax,
        alpha_minus=alpha_minus,
        alpha_plus=alpha_plus,
    )
    xinit = jnp.zeros((dim,))
    vinit = jnp.ones((dim,))
    begin = time()
    out = sampler.sample_skeleton(100000, xinit, vinit, seed, verbose=False)
    out.x.block_until_ready()
    end = time()
    dico = out._asdict()
    dico["grid_size"] = grid_size
    dico["tmax"] = tmax
    dico["ratio_alphas"] = ratio_alphas
    dico["magnitude_alphas"] = magnitude_alphas
    dico["time"] = end - begin
    dico.pop("x")
    dico.pop("v")
    dico.pop("t")
    dico.pop("horizon")
    dico["hitting_horizon"] = np.sum(out.hitting_horizon)
    dico["rejected"] = np.sum(out.rejected)

    return dico


n_rep = 10
gen = np.random.default_rng(1)
seeds = gen.integers(0, 10000, n_rep)
grid_sizes = [0]
tmaxs = [0.0]
ratios_alphas = [0.1, 0.2, 0.5, 1, 2, 5, 10]
magnitudes_alphas = [0.01]
iterable2 = list(product(grid_sizes, seeds, tmaxs, ratios_alphas, magnitudes_alphas))
iterable = iterable2
gen.shuffle(iterable)  # type: ignore
print(len(iterable))
results = Parallel(n_jobs=8, backend="threading", verbose=0)(
    delayed(loop)(*args) for args in tqdm(iterable)
)
# %%
df = pd.DataFrame(results)


def convert_jax_to_numpy(value):
    if isinstance(value, jnp.ndarray):
        return np.array(value)
    return value


for col in df.columns:
    df[col] = df[col].apply(convert_jax_to_numpy)  # type: ignore
# df.to_pickle("high_dimension.zip")

# %%
table = (
    df.groupby(["magnitude_alphas", "ratio_alphas"])[
        ["time", "rejected", "hitting_horizon"]
    ]
    .mean()
    .reset_index()
)
table["horizon_hitted_over_rejections"] = table["hitting_horizon"] / table["rejected"]
table["inverse_ratio"] = 1 / table["ratio_alphas"]
print(table)
table.to_latex(index=False, escape=False, column_format="l" * 4)
# %%
sns.relplot(
    table,
    x="inverse_ratio",
    y="horizon_hitted_over_rejections",
    hue="magnitude_alphas",
    kind="scatter",
)
# %%
x = np.log(table["ratio_alphas"])
y = np.log(table["horizon_hitted_over_rejections"])
sns.regplot(x=x, y=y)

# plot x=y line
coefficients = np.polyfit(x, y, 1)
r2 = np.corrcoef(x, y)[0, 1] ** 2
plt.title(
    f"log(# of hit horizons/# of rejected) = {coefficients[0]:.2f} log(ratio_alphas) + {coefficients[1]:.2f}"
)
plt.show()
# %%
