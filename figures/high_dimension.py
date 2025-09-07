# %%
import os

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


@jax.jit
def true_mean(out):
    mean = (out.x[1:] + out.x[:-1]) / 2 * (out.t[1:] - out.t[:-1])[:, None]
    return jnp.sum(mean, axis=0) / (out.t[-1] - out.t[0])


dim = 300


def U(x):
    mean_x2 = x[0] ** 2 - 1
    return -(-(x[0] ** 2) + -((x[1] - mean_x2) ** 2) - jnp.sum((x[2:]) ** 2)) / 2


def U_2(x, y):
    arr = jnp.zeros(dim)
    arr = arr.at[0].set(x)
    arr = arr.at[1].set(y)
    return U(arr)


# def U(x):
#     x_begin = x[:-1]
#     x_end = x[1:]
#     mean_x2 = (x_begin[0]**2 - 1)
#     return jnp.sum((x_begin - 1)**2 + 100*(x_end - x_begin**2)**2)/2

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
seed = 9
xinit = -jnp.zeros((dim,)) + 1
vinit = jnp.ones((dim,))
grid_size = 3
tmax = 2.5
sampler = pdmp.ForwardEventChain(
    dim, grad_U, grid_size, tmax, vectorized_bound=False, signed_bound=True
)
out = sampler.sample_skeleton(1000000, xinit, vinit, seed, verbose=True)
sample = sampler.sample_from_skeleton(1000000, out)
error = true_mean(out)  # - jnp.mean(means_gauss,axis=0)
pdmp.plot(out)
print(jnp.linalg.norm(error))

ar_reject = jnp.concatenate(out.error_value_ar)
# drop zeros
ar_reject = ar_reject[ar_reject != 0]
sns.histplot(ar_reject)  # type: ignore
sns.jointplot(x=sample[:, 0], y=sample[:, 1])
# plot ess
plt.show()


# %%
def loop(grid_size, seed, tmax, signed):
    sampler = pdmp.ForwardEventChain(
        dim, grad_U, grid_size, tmax, vectorized_bound=False, signed_bound=signed
    )
    xinit = jnp.zeros((dim,))
    vinit = jnp.ones((dim,))
    begin = time()
    out = sampler.sample_skeleton(1000000, xinit, vinit, seed, verbose=False)
    out.x.block_until_ready()
    end = time()
    dico = out._asdict()
    dico["grid_size"] = grid_size
    dico["tmax"] = tmax
    dico["signed"] = signed
    dico["time"] = end - begin
    true_mean_ = true_mean(out)
    sample = sampler.sample_from_skeleton(1000000, out)
    dico["mean_error"] = np.array(true_mean_)
    dico["larger_fiveteen"] = np.sum(sample[:, 1] > 15)
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
grid_sizes = [3, 5, 10, 20]
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
gen = np.random.default_rng()
gen.shuffle(iterable)  # type: ignore
print(len(iterable))
results = Parallel(n_jobs=4, backend="threading", verbose=0)(
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
df["ar_mean"] = df["ar"].apply(lambda x: np.mean(x)).astype(float)
df["rejected_sum"] = df["rejected"].apply(lambda x: np.sum(x)).astype(float)
df["upper_bound_evals"] = df["hitting_horizon"] + 100000
df["gradient_evals"] = (
    df["upper_bound_evals"] * df["grid_size"] + df["rejected_sum"] + 100000
)
df["mse"] = df["mean_error"].apply(lambda x: np.mean(x**2)).astype(float)
# concatenate vect and signed into a single column
# df["vect_signed"] = df["vect"].astype(str) +' '+ df["signed"].astype(str)# %%
df["mean_1"] = df["mean_error"].apply(lambda x: x[1]).astype(float)
df["ar_reject_mean"] = df["ar_reject"].apply(lambda x: np.mean(x - 1)).astype(float)
df["horizon_mean"] = df["horizon"].apply(lambda x: np.mean(x)).astype(float)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))

sns.boxplot(data=df, x="grid_size", y="time", hue="signed", ax=ax[0])
# ax[0].set_yscale("log")
ax[0].set_xlabel("# of grid points")
ax[0].set_ylabel("# number of error bound")
ax[0].legend(title="Vectorized / signed")
ax[0].set_title("Number of errors in the bound")

sns.boxplot(data=df, x="grid_size", y="larger_fiveteen", hue="signed", ax=ax[1])
# ax[1].set_yscale("log")
ax[1].set_xlabel("# of grid points")
ax[1].set_ylabel("Mean of acceptance rate when larger than 1")
ax[1].legend(title="Vectorized / signed")
ax[1].set_title("Measure of the error in the acceptance rate")
# plt.hlines(0,-1,5)
# %%
df.groupby(["grid_size", "signed"])["error_bound"].describe()
# %%
sns.set_theme(context="paper", style="ticks")
sns.barplot(data=df, x="grid_size", y="error_bound", hue="signed")
plt.xlabel("# of grid points")
plt.ylabel("Time (s)")
# plt.savefig("time_high_dimension.pdf")
# %%
table = df.groupby(["grid_size", "signed"])[
    ["time", "horizon_mean", "ar_mean", "rejected_sum", "hitting_horizon"]
].mean()
table.to_latex(index=False, escape=False, column_format="l" * 4)
# %%
table_df = table.reset_index()

# Round numeric columns to 2 decimal places
numeric_columns = table_df.select_dtypes(include=[np.number]).columns
table_df[numeric_columns] = table_df[numeric_columns].round(2)

# Convert to LaTeX
latex_table = table_df.to_latex(
    index=False, escape=False, column_format="l" * len(table_df.columns)
)

# Print the LaTeX code
print(latex_table)
# %%
# Assuming your grouped DataFrame is called 'grouped_df'
table_df = table.reset_index()

# Round numeric columns to 2 decimal places
numeric_columns = table_df.select_dtypes(include=[np.number]).columns
# table_df[numeric_columns] = table_df[numeric_columns].round(4)

# Create a list of column alignments
alignments = ["l"] * len(table_df.columns)
for i, col in enumerate(table_df.columns):
    if col in numeric_columns:
        alignments[i] = "r"  # right-align numeric columns

# Convert to LaTeX
latex_table = table_df.to_latex(
    index=False,
    escape=False,
    column_format="".join(alignments),
    float_format="{:#0.3g}".format,
    caption="Your table caption here",
    label="tab:your_label_here",
    position="htbp",
    bold_rows=True,
    longtable=False,
)

# Add booktabs
latex_table = latex_table.replace(
    "\\begin{table}", "\\begin{table}[htbp]\n\\usepackage{booktabs}"
)
latex_table = latex_table.replace("\\toprule", "\\toprule\n")
latex_table = latex_table.replace("\\midrule", "\\midrule\n")
latex_table = latex_table.replace("\\bottomrule", "\\bottomrule\n")

print(latex_table)
# %%
table
# %%
