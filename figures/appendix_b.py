# %%
from __future__ import annotations

import os

os.environ["XLA_FLAGS"] = "--xla_cpu_use_thunk_runtime=false"

from itertools import product
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import pdmp_jax as pdmp
from pdmp_jax.utils import alpha_minus_plus_from_ratio_and_magnitude

if TYPE_CHECKING:
    from jaxtyping import Array

n_gauss = 20
dim = 2
# means_gauss = jax.random.uniform(jax.random.PRNGKey(3), shape=(n_gauss, dim),minval=-1.,maxval=1.)  *3
means_gauss = jax.random.normal(jax.random.PRNGKey(1), shape=(n_gauss, dim)) * 3


def U(x: Array) -> Array:
    return -jax.nn.logsumexp(-jnp.sum((x - means_gauss) ** 2, axis=-1) / 2)


# %%
grad_U = jax.grad(U)
seed = 42
xinit = -jnp.zeros((dim,))
vinit = jnp.ones((dim,))
grid_size = 20
tmax = 3.5

ratios_alphas = [0.2, 1, 5]
magnitudes_alphas = [0.0025, 0.01, 0.04]

dico = {}
for ratio, magnitude in product(ratios_alphas, magnitudes_alphas):
    alpha_minus, alpha_plus = alpha_minus_plus_from_ratio_and_magnitude(
        ratio, magnitude
    )
    sampler = pdmp.ZigZag(
        dim, grad_U, grid_size, tmax, alpha_minus=alpha_minus, alpha_plus=alpha_plus
    )
    out = sampler.sample_skeleton(2000, xinit, vinit, seed, verbose=False)

    dico[(ratio, magnitude)] = out.horizon

# %%
plt.figure(figsize=(12, 8))
# Alternative approach using named colormaps
cmaps = {0.2: plt.get_cmap("Reds"), 1: plt.get_cmap("Greens"), 5: plt.get_cmap("Blues")}

for (ratio, magnitude), horizon in dico.items():
    hue = ratios_alphas.index(ratio) / len(ratios_alphas) * 0.8

    color = cmaps[ratio](
        0.3 + 0.7 * (magnitudes_alphas.index(magnitude) / len(magnitudes_alphas))
    )
    plt.plot(horizon, label=f"R: {ratio}, M: {magnitude}", color=color, lw=1.5)


plt.xlabel("Step")
plt.ylabel("Horizon")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Parameters")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
