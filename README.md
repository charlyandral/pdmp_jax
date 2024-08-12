# pdmp_jax


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

