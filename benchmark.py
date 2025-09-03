import jax

jax.config.update("jax_enable_x64", True)
# NB: this is not necessary, but it is useful to avoid numerical issues. It must be done before importing pdmp_jax.

from time import time

import jax.numpy as jnp

import pdmp_jax as pdmp


def U(x):
    return jnp.sum(x**2)


dim = 50
# define the gradient of the potential
grad_U = jax.grad(U)
seed = 8
key = jax.random.PRNGKey(seed)
xinit = jnp.ones((dim,))  # initial position
vinit = jnp.ones((dim,))  # initial velocity
grid_size = 10
N = 1000000  # number of samples
sampler = pdmp.Boomerang(dim, grad_U, grid_size)
sampler.sample(1, 1, xinit, vinit, seed, verbose=False)

start = time()
sample = sampler.sample(
    N_sk=N, N_samples=N, xinit=xinit, vinit=vinit, seed=seed, verbose=False
).block_until_ready()
end = time()

print(f"Time: {end - start}")
