import jax 
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pdmp_jax as pdmp

@jax.jit
def true_mean(out):
    mean = (out.x[1:] + out.x[:-1])/2 * (out.t[1:] - out.t[:-1])[:,None]
    return jnp.sum(mean,axis=0) / (out.t[-1] - out.t[0])


dim = 50

def U(x):
    mean_x2 = (x[0]**2 - 1 )
    return -(- x[0]**2 + -(x[1]-mean_x2)**2 - jnp.sum((x[2:])**2) )/ 2


def U_2(x,y):
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
# plt.figure()
# plt.contour(X, Y, Z, levels=200)
# plt.show()

plt.figure()
grad_U = jax.grad(U)
seed = 9
xinit = jnp.ones((dim,)) 
vinit = jnp.ones((dim,))
grid_size = 3
tmax = 2.5
sampler = pdmp.ForwardEventChain(dim, grad_U, grid_size, tmax,vectorized_bound=False,signed_bound=True)
out = sampler.sample_skeleton(1000000, xinit, vinit, seed,verbose = True)
sample = sampler.sample_from_skeleton(1000000,out)
error = true_mean(out) #- jnp.mean(means_gauss,axis=0)
pdmp.plot(out)
plt.show()

# ar_reject = jnp.concatenate(out.error_value_ar)
# # drop zeros
# ar_reject = ar_reject[ar_reject != 0]
# sns.histplot(ar_reject)
plt.figure()
sns.jointplot(x = sample[:,0],y = sample[:,1])
plt.show()
