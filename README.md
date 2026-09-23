# pdmp_jax

## ⚠️ Performance Notice

Since JAX 0.4.32, XLA uses a new CPU runtime (the "thunk" runtime) that adds a fixed cost every time the body of a `while_loop` or a `lax.cond` branch is executed. The sampling loop is made of such small, data-dependent loops, so with default settings it runs about 3-4x slower in low and moderate dimension (the gap vanishes in high dimension, where the rate evaluations dominate), and about 40x slower on JAX 0.7.1. Set one of the following before importing JAX:

- **JAX ≥ 0.7.1** (tested up to 0.11.2): let XLA compile the while loops into single kernels by raising the size limit of its small-while-loop pass (1 kB by default):

  ```bash
  export XLA_FLAGS="--xla_backend_extra_options=xla_cpu_small_while_loop_byte_threshold=1000000000"
  ```

- **JAX 0.4.32 to 0.6.2** (0.6.2 is pinned in `pyproject.toml`): use the old runtime. This flag is ignored from 0.7 on, and makes 0.7.0 and 0.7.1 crash (`CpuExecutable has no thunks`).

  ```bash
  export XLA_FLAGS="--xla_cpu_use_thunk_runtime=false"
  ```

Both restore the same speed, and give the same results up to floating-point rounding (bit-identical with a grid bound; with the Brent bound, `grid_size=0`, the first setting reproduces the old runtime exactly, whereas the default thunk runtime rounds differently and uses about 2% more rate evaluations). The small-while-loop pass skips loops that contain `sort`, `scatter`, FFT or custom calls, so a potential using e.g. `jnp.sort` does not benefit from it.

Indicative throughput (ZigZag, Gaussian target, dimension 50, grid size 10, Apple Silicon CPU):

| JAX version | `XLA_FLAGS` | events/s |
|---|---|---|
| 0.6.2 | none | ~110k |
| 0.6.2 | `--xla_cpu_use_thunk_runtime=false` | ~420k |
| 0.7.1 | none | ~2k |
| 0.7.1 | `--xla_backend_extra_options=xla_cpu_small_while_loop_byte_threshold=1000000000` | ~410k |
| 0.11.2 | none | ~120k |
| 0.11.2 | `--xla_backend_extra_options=xla_cpu_small_while_loop_byte_threshold=1000000000` | ~415k |

## Documentation
This repository contains a JAX implementation of the PDMP sampler describe in the article ["Automated Techniques for Efficient Sampling of Piecewise-Deterministic Markov Processes"](https://arxiv.org/abs/2408.03682).
The following PDMP samplers are implemented:
- [Zig-Zag sampler](https://doi.org/10.1214/18-AOS1715) (Joris Bierkens, Paul Fearnhead, Gareth Roberts. "The Zig-Zag process and super-efficient sampling for Bayesian analysis of big data." The Annals of Statistics, 47(3) 1288-1320 June 2019.)
- [Bouncy Particle Sampler](https://doi.org/10.1080/01621459.2017.1294075) (Bouchard-Côté, A., Vollmer, S. J., & Doucet, A. (2018). The Bouncy Particle Sampler: A Nonreversible Rejection-Free Markov Chain Monte Carlo Method. Journal of the American Statistical Association, 113(522), 855–867. )
- [Forward Event Chain](https://doi.org/10.1080/10618600.2020.1750417) (Forward Ref with random time for the orthogonal switch) (Michel, M., Durmus, A., & Sénécal, S. (2020). Forward Event-Chain Monte Carlo: Fast Sampling by Randomness Control in Irreversible Markov Chains. Journal of Computational and Graphical Statistics, 29(4), 689–702.)
- [Speedup ZigZag](https://doi.org/10.1214/23-AAP1930) (non explosive case) (G. Vasdekis, G. O. Roberts. "Speed up Zig-Zag." The Annals of Applied Probability, 33(6A) 4693-4746 December 2023. )
- [Boomerang Sampler](https://proceedings.mlr.press/v119/bierkens20a.html ) (Bierkens, J., Grazzi, S., Kamatani, K. &amp; Roberts, G.. (2020). The Boomerang Sampler. <i>Proceedings of the 37th International Conference on Machine Learning</i>, in <i>Proceedings of Machine Learning Research</i> 119:908-918)



It can be installed using pip:
```bash
pip install pdmp-jax
```




## Example
    
```python
import jax 
jax.config.update("jax_enable_x64", True)


import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pdmp_jax as pdmp



# define a potential 
# here 2D banana potential and gaussian on the other dimensions
def U(x):
    mean_x2 = (x[0]**2 - 1 )
    return -(- x[0]**2 + -(x[1]-mean_x2)**2 - jnp.sum((x[2:])**2) )/ 2

dim = 50
# define the gradient of the potential. Using JAX, no need to define it explicitly
grad_U = jax.grad(U)
seed = 8
key = jax.random.PRNGKey(seed)
xinit = jnp.ones((dim,)) # initial position
vinit = jnp.ones((dim,))  # initial velocity
grid_size = 10 # number of grid points
N_sk = 1000000 # number of skeleton points
N = 1000000 # number of samples
sampler = pdmp.ZigZag(dim, grad_U, grid_size)
# sample the skeleton of the process 
out = sampler.sample_skeleton(N_sk, xinit, vinit, seed,verbose = True)

# sample from the skeleton
sample = sampler.sample_from_skeleton(N,out)

# other possibilty : use sample() method directly
sample2 = sampler.sample(N_sk=N_sk, N_samples=N, xinit=xinit, vinit=vinit, seed=seed, verbose=True)

# plot the first two dimensions of the sample
plt.figure()
sns.jointplot(x = sample[:,0],y = sample[:,1])
plt.show()
```


The file `example.ipynb` contains a more detailed example with all the different PDMP samplers implemented in the package.

## A few remarks

- The package is built on top of JAX, so the potential should be defined using JAX functions to benefit from the automatic differentiation.
- The package is built to be as general as possible, so it should be easy to add new PDMP samplers by defining a new class that inherits from the PDMP class.
- The implementation of the main sampling loop is done using the 'jax.lax.scan' function, thus it uses JAX's JIT compiler.
- All tests were done on a CPU, so even if JAX is GPU compatible, the package has not been tested on a GPU.
- The package is still under development, so if you find any bugs or have any suggestions, please open an issue or a pull request.

