# %%
import os

os.environ["XLA_FLAGS"] = "--xla_cpu_use_thunk_runtime=false"

from itertools import product

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax.scipy.special import logsumexp
from joblib import Parallel, delayed

import pdmp_jax as pdmp

# %%
# Figures 1 and 2 of the paper: effect of tmax on the number of gradient
# evaluations for the ZigZag sampler with the Corbella-style early-stopping
# constant bound (grid_size=0 + early_stop_bound). Setup of the original runs
# (analysis_tmax.py): dim=5 mixture of two Gaussians, 10000 skeleton points.
dim = 30
inv_var_2 = 10
weights = jnp.array([1.0, inv_var_2 ** (dim / 2)])


def U(x):
    x_shift = x - 1.0
    return -logsumexp(
        -jnp.array([x @ x / 2, x_shift @ (inv_var_2 * x_shift) / 2]), b=weights
    )


# grad_U = jax.grad(U)
grad_U = lambda x: x
n_sk = 10000


def gradient_evals(tmax, adaptive, seed):
    sampler = pdmp.ZigZag(
        dim,
        grad_U,
        grid_size=0,
        tmax=float(tmax),
        adaptive=adaptive,
        alpha_minus=1.1,
        alpha_plus=1.1,
        early_stop_bound=True,
    )
    gen = np.random.RandomState(seed)
    xinit = jnp.asarray(gen.randn(dim))
    vinit = jnp.asarray(gen.randint(0, 2, dim) * 2.0 - 1.0)
    out = sampler.sample_skeleton(n_sk, xinit, vinit, seed, verbose=False)
    c_opt = int(np.sum(out.bound_evals))
    c_tpp = n_sk + int(np.sum(out.rejected)) + int(np.sum(out.error_bound))
    return c_opt, c_tpp


def sweep(tmaxs, adaptive, n_rep=4, n_jobs=6):
    jobs = list(product(tmaxs, range(n_rep)))
    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(gradient_evals)(tm, adaptive, s) for tm, s in jobs
    )
    return np.array(results).reshape(len(tmaxs), n_rep, 2).mean(axis=1)


def plot_counts(tmaxs, counts):
    c_opt, c_tpp = counts[:, 0], counts[:, 1]
    plt.figure(figsize=(6, 4))
    plt.semilogx(tmaxs, c_opt, label="$C^{opt}$", linewidth=1)
    plt.semilogx(tmaxs, c_tpp, label="$C^{tpp}$", linewidth=1)
    plt.semilogx(tmaxs, c_opt + c_tpp, label="$C^{tot}$", linewidth=1)
    plt.xlabel("tmax")
    plt.legend()


# %%
# Figure 1: non-adaptive (tmax fixed for the whole run)
tmaxs_fixed = np.logspace(-2, 1, 15)
counts_fixed = sweep(tmaxs_fixed, adaptive=False)
plot_counts(tmaxs_fixed, counts_fixed)
# plt.savefig("effect_tmax_non_adaptive.pdf", bbox_inches="tight")
plt.show()

# %%
# Figure 2: adaptive tmax with alpha = 1.1 (paper used 12 repetitions)
tmaxs_adaptive = np.logspace(-4, 3, 15)
counts_adaptive = sweep(tmaxs_adaptive, adaptive=True)
plot_counts(tmaxs_adaptive, counts_adaptive)
# plt.savefig("effect_tmax_adaptive.pdf", bbox_inches="tight")
plt.show()

# %%
