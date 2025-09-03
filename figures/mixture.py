# %%
import os

os.environ["XLA_FLAGS"] = "--xla_cpu_use_thunk_runtime=false"
from itertools import product
from time import time

import jax

print(jax.__version__)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed

import pdmp_jax as pdmp

# %load_ext autoreload
# %autoreload 2
# %%
dim = 2


sd = 0.03
inv_var_2 = 1 / sd**2

weights = jnp.array([1, 1 / (sd**dim)])


# @jax.jit
def U(x):
    # scale = jnp.logspace(0,6,dim)**.5
    # x = x / scale
    x_shift = x - 1.0
    return -jax.nn.logsumexp(
        -jnp.array([x.T @ x / 2, x_shift.T @ (inv_var_2 * x_shift) / 2]), b=weights
    )


def U_2(x, y):
    return U(jnp.array([x, y]))


U_vect = jax.vmap(jax.vmap(U_2))
grad_U = jax.grad(U)
# make a 2D plot of the potential
x = np.linspace(-10, 10, 200)
y = np.linspace(-10, 10, 200)
X, Y = np.meshgrid(x, y)
Z = U_vect(X, Y)

plt.contour(X, Y, Z, levels=200)
# compute true mean from skeleton


@jax.jit
def true_mean(out):
    mean = (out.x[1:] + out.x[:-1]) / 2 * (out.t[1:] - out.t[:-1])[:, None]
    return jnp.sum(mean, axis=0) / (out.t[-1] - out.t[0])


# %%
grid_size = 30
tmax = 0.1
sampler = pdmp.BouncyParticle(dim, grad_U, grid_size, tmax, adaptive=True)
seed = 10
seed1, seed2 = jax.random.split(jax.random.PRNGKey(seed))
xinit = jax.random.normal(seed1, shape=(2,))
# vinit = jax.random.randint(seed2, (2,), 0, 2) * 2 - 1
vinit = jax.random.normal(seed2, shape=(2,))
vinit = vinit / jnp.linalg.norm(vinit)
out = sampler.sample_skeleton(1000000, xinit, vinit, seed, verbose=True)
sample = sampler.sample_from_skeleton(100000, out)
pdmp.plot(out)

print(true_mean(out))

plt.show()
plt.plot(out.horizon)
# %%


def loop(grid_size, seed, tmax):
    sampler = pdmp.BouncyParticle(dim, grad_U, grid_size, tmax, adaptive=False)
    seed1, seed2 = jax.random.split(jax.random.PRNGKey(seed))
    xinit = jax.random.normal(seed1, shape=(2,))
    vinit = jax.random.normal(seed2, shape=(2,))
    vinit = vinit / jnp.linalg.norm(vinit)
    begin = time()
    out = sampler.sample_skeleton(1000000, xinit, vinit, seed, verbose=False)
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
seeds = np.arange(n_seed)
tmaxs = [0.0, 0.01, 0.1, 1.0]
# tmaxs = [0.0]
# vects_signeds = [(False,False),(True,False),(True,True)]
# vects_signeds = [(True,True)]
iterable = list(product(grid_sizes, seeds, tmaxs))

from tqdm.notebook import tqdm

np.random.shuffle(iterable)
print(len(iterable))
results_fec = Parallel(n_jobs=8, backend="loky")(
    delayed(loop)(*args) for args in tqdm(iterable)
)


# %%
df = pd.DataFrame(results_fec)
df["mean"] = df["mean_error"] + 0.5
df["mean_1"] = df["mean"].apply(lambda x: x[0]).astype(float)
df["mean_2"] = df["mean"].apply(lambda x: x[1]).astype(float)
# %%
# plot the mean_1 and mean_2 as hue of the violin plot
df_melt = pd.melt(
    df,
    id_vars=["grid_size"],
    value_vars=["mean_1", "mean_2"],
    var_name="marginal",
    value_name="mean_value",
)

sns.set_theme(context="paper", style="ticks")
sns.boxplot(data=df, x="grid_size", y="mean_1", hue="tmax")
plt.axhline(y=0.5, color="r", linestyle="--")
plt.legend(title="tmax", loc="lower right")
plt.ylabel("Mean of the first marginal")
plt.xlabel("# of grid points")
# plt.savefig("2mixture_mean.pdf")
# %%
sns.set_theme(context="paper", style="ticks")
sns.barplot(data=df, x="grid_size", y="time", hue="tmax")
plt.ylabel("Time (s)")
plt.xlabel("# of grid points")
# plt.savefig("2mixture_time.pdf")
# plt.semilogy()
# %%
