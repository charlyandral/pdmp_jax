# %%
import os

os.environ["XLA_FLAGS"] = "--xla_cpu_use_thunk_runtime=false"

from itertools import product
from time import time
from typing import Any, overload

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from tqdm.notebook import tqdm

import pdmp_jax as pdmp


@jax.jit
def true_mean(out):
    mean = (out.x[1:] + out.x[:-1]) / 2 * (out.t[1:] - out.t[:-1])[:, None]
    return jnp.sum(mean, axis=0) / (out.t[-1] - out.t[0])


@overload
def convert_jax_to_numpy(value: jnp.ndarray) -> np.ndarray: ...
@overload
def convert_jax_to_numpy(value: Any) -> Any: ...
def convert_jax_to_numpy(value: jnp.ndarray | Any) -> Any:
    if isinstance(value, jnp.ndarray):
        return np.array(value)
    return value


# %%
n_gauss = 20
dim = 2
# means_gauss = jax.random.uniform(jax.random.PRNGKey(3), shape=(n_gauss, dim),minval=-1.,maxval=1.)  *3
means_gauss = jax.random.normal(jax.random.PRNGKey(1), shape=(n_gauss, dim)) * 3


def U(x):
    return -jax.nn.logsumexp(-jnp.sum((x - means_gauss) ** 2, axis=-1) / 2)


def U_2(x, y):
    arr = jnp.zeros(dim)
    arr = arr.at[0].set(x)
    arr = arr.at[1].set(y)
    return U(arr)


U_vect = jax.vmap(jax.vmap(U_2))
grad_U = jax.grad(U)
# make a 2D plot of the potential
x = np.linspace(-10, 10, 200)
y = np.linspace(-10, 10, 200)
X, Y = np.meshgrid(x, y)
Z = U_vect(X, Y)

plt.contour(X, Y, Z, levels=200)
# %%

grad_U = jax.grad(U)
seed = 10
xinit = -jnp.zeros((dim,))
vinit = jnp.ones((dim,))
grid_size = 20
tmax = 2.5
sampler = pdmp.ZigZag(dim, grad_U, grid_size, tmax, alpha_minus=1.1, alpha_plus=1.1)
out = sampler.sample_skeleton(200000, xinit, vinit, seed, verbose=True)
# sample = sampler.sample_from_skeleton(1000000,out)
error = true_mean(out) - jnp.mean(means_gauss, axis=0)
pdmp.plot(out)
print(jnp.linalg.norm(error))

ar_reject = jnp.concatenate(out.error_value_ar)
# drop zeros
ar_reject = ar_reject[ar_reject != 0]
sns.histplot(ar_reject)  # type: ignore
# %%
plt.plot(out.horizon)
# %%


def loop_zz(grid_size, seed, tmax, vect_signed) -> dict[str, Any]:
    vect = vect_signed[0]
    signed = vect_signed[1]
    sampler = pdmp.ZigZag(
        dim, grad_U, grid_size, tmax, vectorized_bound=vect, signed_bound=signed
    )
    begin = time()
    out = sampler.sample_skeleton(1000000, xinit, vinit, seed, verbose=False)
    out.x.block_until_ready()
    end = time()
    dico = out._asdict()
    dico["grid_size"] = grid_size
    dico["tmax"] = tmax
    dico["vect"] = vect
    dico["signed"] = signed
    dico["time"] = end - begin
    true_mean_ = true_mean(out)
    dico["mean_error"] = np.array(true_mean_ - jnp.mean(means_gauss, axis=0))
    dico.pop("x")
    dico.pop("v")
    dico.pop("t")
    dico["error_bound"] = np.sum(out.error_bound)
    dico["hitting_horizon"] = np.sum(out.hitting_horizon)
    ar_reject = jnp.concatenate(out.error_value_ar)
    dico["ar_reject"] = ar_reject[ar_reject != 0.0]
    dico.pop("error_value_ar")
    dico["rejected"] = np.sum(out.rejected)
    dico["ar"] = np.mean(out.ar)

    return dico


# for non constant bound
grid_sizes = [5, 10, 20, 50]
seeds = np.arange(20)
tmaxs = [0.0]
vects_signeds = [(False, False), (True, False), (True, True)]
iterable1 = list(product(grid_sizes, seeds, tmaxs, vects_signeds))
# for constant bound
grid_sizes = [0]
tmaxs = [0.0]
vects_signeds = [(False, False)]
iterable2 = list(product(grid_sizes, seeds, tmaxs, vects_signeds))
iterable = iterable1 + iterable2

np.random.shuffle(iterable)  # type: ignore
print(len(iterable))
results_boomerang: list[dict] = Parallel(n_jobs=8, backend="loky", verbose=10)(
    delayed(loop_zz)(*args) for args in tqdm(iterable)
)
# %%
results_zz = results_boomerang
df = pd.DataFrame(results_zz)


for col in df.columns:
    df[col] = df[col].apply(convert_jax_to_numpy)
# %%

# %%
# df.to_pickle("results_zz_new.zip")
# %%
# df = pd.read_pickle("results_zz_new.zip")
df["ar_mean"] = df["ar"].apply(lambda x: np.mean(x)).astype(float)
df["rejected_sum"] = df["rejected"].apply(lambda x: np.sum(x)).astype(float)
df["upper_bound_evals"] = df["hitting_horizon"] + 100000
df["gradient_evals"] = (
    df["upper_bound_evals"] * df["grid_size"] + df["rejected_sum"] + 100000
)
df["mse"] = df["mean_error"].apply(lambda x: np.mean(x**2)).astype(float)
# concatenate vect and signed into a single column
df["vect_signed"] = df["vect"].astype(str) + " " + df["signed"].astype(str)  # %%
df["mean_1"] = df["mean_error"].apply(lambda x: x[1]).astype(float)
df["ar_reject_mean"] = df["ar_reject"].apply(lambda x: np.mean(x - 1)).astype(float)
df["horizon_mean"] = df["horizon"].apply(lambda x: np.mean(x)).astype(float)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))

sns.boxplot(data=df, x="grid_size", y="error_bound", hue="vect_signed", ax=ax[0])
ax[0].set_yscale("log")
ax[0].set_xlabel("# of grid points")
ax[0].set_ylabel("# number of error bound")
ax[0].legend(title="Vectorized / signed")
ax[0].set_title("Number of errors in the bound")

sns.boxplot(data=df, x="grid_size", y="ar_reject_mean", hue="vect_signed", ax=ax[1])
ax[1].set_yscale("log")
ax[1].set_xlabel("# of grid points")
ax[1].set_ylabel("Mean of acceptance rate when larger than 1")
ax[1].legend(title="Vectorized / signed")
ax[1].set_title("Measure of the error in the acceptance rate")
# plt.hlines(0,-1,5)
plt.savefig("error_bound_ar_reject_zz.pdf")

# %%


def loop_boomerang(grid_size, seed, tmax, signed) -> dict[str, Any]:
    sampler = pdmp.Boomerang(
        dim, grad_U, grid_size, tmax, vectorized_bound=False, signed_bound=signed
    )
    begin = time()
    xinit = jnp.zeros((dim,))
    vinit = jnp.ones((dim,))
    out = sampler.sample_skeleton(1000000, xinit, vinit, seed, verbose=False)
    out.x.block_until_ready()
    end = time()
    dico = out._asdict()
    dico["grid_size"] = grid_size
    dico["tmax"] = tmax
    dico["signed"] = signed
    dico["time"] = end - begin
    true_mean_ = true_mean(out)
    dico["mean_error"] = np.array(true_mean_ - jnp.mean(means_gauss, axis=0))
    dico.pop("x")
    dico.pop("v")
    dico.pop("t")
    dico["error_bound"] = np.sum(out.error_bound)
    dico["hitting_horizon"] = np.sum(out.hitting_horizon)
    ar_reject = jnp.concatenate(out.error_value_ar)
    dico["ar_reject"] = ar_reject[ar_reject != 0.0]
    dico.pop("error_value_ar")
    dico["rejected"] = np.sum(out.rejected)
    dico["ar"] = np.mean(out.ar)

    return dico


# for non constant bound
grid_sizes = [5, 10, 20, 50]
seeds = np.arange(20)
tmaxs = [0.0]
signeds = [True, False]
iterable1 = list(product(grid_sizes, seeds, tmaxs, signeds))
# for constant bound
grid_sizes = [0]
tmaxs = [0.0]
signeds = [False]
iterable2 = list(product(grid_sizes, seeds, tmaxs, signeds))
iterable = iterable1 + iterable2
np.random.shuffle(iterable)  # type: ignore
print(len(iterable))
results_boomerang = Parallel(n_jobs=8, backend="loky", verbose=0)(
    delayed(loop_boomerang)(*args) for args in tqdm(iterable)
)
# %%
df_boom = pd.DataFrame(results_boomerang)


for col in df_boom.columns:
    df_boom[col] = df_boom[col].apply(convert_jax_to_numpy)

df_boom["ar_reject_mean"] = (
    df_boom["ar_reject"].apply(lambda x: np.mean(x - 1)).astype(float)
)

# df_boom.to_pickle("results_boom_new.zip")


fig, ax = plt.subplots(1, 2, figsize=(10, 5))

sns.boxplot(data=df_boom, x="grid_size", y="error_bound", hue="signed", ax=ax[0])
# ax[0].set_yscale("log")
ax[0].set_xlabel("# of grid points")
ax[0].set_ylabel("# number of error bound")
ax[0].legend(title="Signed")
ax[0].set_title("Number of errors in the bound")

sns.boxplot(data=df_boom, x="grid_size", y="ar_reject_mean", hue="signed", ax=ax[1])
ax[1].set_yscale("log")
ax[1].set_xlabel("# of grid points")
ax[1].set_ylabel("Mean of acceptance rate when larger than 1")
ax[1].legend(title="Signed")
ax[1].set_title("Measure of the error in the acceptance rate")
# plt.hlines(0,-1,5)
# ror_bound_ar_reject_boomerang.pdf")
# %%
