import jax
import jax.numpy as jnp
from jax.tree_util import Partial as jax_partial

from typing import  Callable, Tuple
from jaxtyping import Float, Int, PRNGKeyArray, Bool, Array

from jax_tqdm import scan_tqdm

from ..namedtuples import PdmpState, PdmpOutput
from ..sampling_loop import one_step, output_state
from ..upper_bound import upper_bound_constant, upper_bound_grid, upper_bound_grid_vect

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns


class PDMP:
    def __init__(self):
        self.dim: Int
        self.refresh_rate: Float
        self.grad_U: Callable[[Array], Array]
        self.grid_size: Int
        self.tmax: Float

        self.vectorized_bound: Bool
        self.signed_bound: Bool
        self.adaptive: Bool

        self.integrator: Callable[[Array, Array, Float], Tuple[Array, Array]]
        self.rate: Callable[[Array, Array, Float], Float]
        self.velocity_jump: Callable[[Array, Array, PRNGKeyArray], Array]
        self.state: PdmpState | None

    def init_state(
        self,
        xinit: Float[Array, "dim"],
        vinit: Float[Array, "dim"],
        seed: int,
    ):
        """
        Initializes the state of the PDMP sampler.

        Args:
            xinit (Float[Array, "dim"]): The initial position.
            vinit (Float[Array, "dim"]): The initial velocity.
            seed (int): The seed for random number generation.
            upper_bound_vect (bool, optional): Whether to use vectorized upper bound function. Defaults to False.
            signed_rate (bool, optional): Whether to use signed rate function. Defaults to False.
            adaptive (bool, optional): Whether to use adaptive upper bound. Defaults to False.
            constant_bound (bool, optional): Whether to use constant upper bound. Defaults to False.

        Returns:
            PdmpState: The initialized PDMP state.
        """

        key = jax.random.key(seed)
        if self.signed_bound:
            rate = self.signed_rate
            rate_vect = self.signed_rate_vect
            refresh_rate = self.refresh_rate
        else:
            rate = self.rate
            rate_vect = self.rate_vect
            refresh_rate = 0.0

        if self.grid_size == 0:

            def upper_bound_func(x, v, horizon):
                func = jax_partial(lambda t: self.rate(x, v, t))
                return upper_bound_constant(func, 0.0, horizon)

        elif not self.vectorized_bound:

            def upper_bound_func(x, v, horizon):
                func = jax_partial(lambda t: rate(x, v, t))
                return upper_bound_grid(
                    func, 0.0, horizon, self.grid_size, refresh_rate
                )

        else:

            def upper_bound_func(x, v, horizon):
                func = jax_partial(lambda t: rate_vect(x, v, t))
                return upper_bound_grid_vect(func, 0.0, horizon, self.grid_size)

        upper_bound_func = jax_partial(upper_bound_func)
        boundox = upper_bound_func(xinit, vinit, self.tmax)
        state = PdmpState(
            xinit,
            vinit,
            jnp.array(0.0),
            self.tmax,
            key,  # type: ignore
            self.integrator,
            self.grad_U,
            self.rate,
            self.velocity_jump,
            upper_bound_func,
            upper_bound=boundox,
            adaptive=self.adaptive,
        )
        self.state = state
        return state

    def sample_skeleton(
        self,
        n_sk: int,
        xinit: Float[Array, "dim"],
        vinit: Float[Array, "dim"],
        seed: int,
        verbose=True,
    ) -> PdmpOutput:
        """
        Samples the skeleton of the PDMP model.

        Parameters:
        - n_sk (int): The number of skeleton samples to generate.
        - xinit (jnp.ndarray): The initial position of the particles.
        - vinit (jnp.ndarray): The initial velocity of the particles.
        - seed (int): The seed value for random number generation.
        - verbose (bool): Whether to display progress bar during sampling. Default is True.

        Returns:
        - output: The output state of the sampling process.
        """

        def one_step_inside(state, _):
            state = one_step(state)
            output = output_state(state)
            return state, output

        if verbose:
            one_step_inside = scan_tqdm(n_sk)(one_step_inside)

        initial_state = self.init_state(xinit, vinit, seed)
        initial_output = output_state(initial_state)
        state, output = jax.lax.scan(one_step_inside, initial_state, jnp.arange(n_sk))
        self.state = state
        # concatenate the initial output with the output
        output = jax.tree.map(
            lambda x, y: jnp.insert(x, 0, y, axis=0), output, initial_output
        )
        return output

    def sample_from_skeleton(self, N: int, output: PdmpOutput) -> Float[Array, "N dim"]:
        """
        Samples from the skeleton of the PDMP trajectory.

        Args:
            N (int): The number of samples to generate.
            output (PdmpOutput): The PDMP output containing the trajectory information.

        Returns:
            jnp.ndarray: The sampled points from the PDMP trajectory skeleton.
        """
        x, v, t = output.x, output.v, output.t
        t = t - t[0]
        tm = (t[-1] / N) * jnp.arange(1, N + 1)
        index = jnp.searchsorted(t, tm) - 1
        sample = jax.vmap(self.integrator)(x[index], v[index], tm - t[index])[0]
        return sample

    def sample(
        self,
        N_sk: int,
        N_samples: int,
        xinit: Float[Array, "dim"],
        vinit: Float[Array, "dim"],
        seed: int,
        verbose: bool = True,
    ) -> Float[Array, "N_samples dim"]:
        """
        Samples from the PDMP model.

        Args:
            N_sk (int): Number of skeleton points to generate.
            N_samples (int): Number of final samples to generate from the skeleton.
            xinit (jnp.ndarray): Initial position.
            vinit (jnp.ndarray): Initial velocity.
            seed (int): Seed for random number generation.
            verbose (bool, optional): Whether to print progress information. Defaults to True.

        Returns:
            jnp.ndarray: Array of samples generated from the PDMP model.
        """
        output = self.sample_skeleton(N_sk, xinit, vinit, seed, verbose)
        return self.sample_from_skeleton(N_samples, output)

    def _init_zz_rate(self):
        """
        Initializes the ZZ rate functions.

        Returns:
        - A partial function `_global_rate` that calculates the global rate given the current state.
        - A partial function `_global_rate_vect` that calculates the vectorized global rate given the current state.
        - A placeholder `_signe_rate` (currently set to None).
        - A partial function `_signed_rate_vect` that calculates the vectorized signed rate given the current state.
        """

        def _global_rate(x0, v0, t):
            xt, vt = self.integrator(x0, v0, t)
            return jnp.sum(jnp.maximum(0.0, self.grad_U(xt) * vt))

        def _global_rate_vect(x0, v0, t):
            xt, vt = self.integrator(x0, v0, t)
            return jnp.maximum(0.0, self.grad_U(xt) * vt)

        _signe_rate = None

        def _signed_rate_vect(x0, v0, t):
            xt, vt = self.integrator(x0, v0, t)
            return self.grad_U(xt) * vt

        return (
            jax_partial(_global_rate),
            jax_partial(_global_rate_vect),
            _signe_rate,
            jax_partial(_signed_rate_vect),
        )

    def _init_bps_rate(self):
        """
        Initializes the BPS rate functions.

        Returns:
            A tuple containing the following functions:
            - `_global_rate`: Computes the global rate of the BPS algorithm.
            - `_global_rate_vect`: Vectorized version of `_global_rate` (currently set to None).
            - `_signed_rate`: Computes the signed rate of the BPS algorithm.
            - `_signed_rate_vect`: Vectorized version of `_signed_rate` (currently set to None).
        """

        def _global_rate(x0, v0, t):
            xt, vt = self.integrator(x0, v0, t)
            return jnp.maximum(0.0, self.grad_U(xt) @ vt) + self.refresh_rate

        _global_rate_vect = None

        def _signed_rate(x0, v0, t):
            xt, vt = self.integrator(x0, v0, t)
            return self.grad_U(xt) @ vt + self.refresh_rate

        _signed_rate_vect = None

        return (
            jax_partial(_global_rate),
            _global_rate_vect,
            jax_partial(_signed_rate),
            _signed_rate_vect,
        )



def plot(output: PdmpOutput):
    """
    Plots various histograms based on the given PdmpOutput object. The histograms include:
    - Time between events histogram (top left)
    - Acceptance rate histogram (top right)
    - Hitting horizon histogram (bottom left)
    - Rejection histogram (bottom right)
    

    Parameters:
    - output (PdmpOutput): The PdmpOutput object containing the data to be plotted.

    Returns:
    - None
    """

    sns.set_style("ticks")
    # change font
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    # Plot 1: Time between events histogram
    sns.histplot(np.diff(output.t), element="step", ax=axs[0, 0])
    axs[0, 0].set_title("Time between events histogram")

    # Plot 2: Acceptance rate histogram
    sns.histplot(output.ar, element="step", ax=axs[0, 1])
    axs[0, 1].set_title("Acceptance rate histogram")
    # make a vertical line for the mean and add a legend
    axs[0, 1].axvline(
        output.ar.mean(),
        color="r",
        linestyle="dashed",
        linewidth=1,
        label=f"Mean : {output.ar.mean():.3f}",
    )
    axs[0, 1].legend()

    # Plot 3: Hitting horizon histogram
    sns.histplot(
        output.hitting_horizon,
        discrete=True,
        shrink=0.8,
        stat="percent",
        element="bars",
        ax=axs[1, 0],
    )
    axs[1, 0].set_yscale("log")
    axs[1, 0].yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda y, _: "{:.16g}".format(y))
    )
    axs[1, 0].set_title(
        f"Hitting horizon histogram, total : {output.hitting_horizon.sum()}"
    )

    # Plot 4: Rejection histogram
    sns.histplot(
        output.rejected,
        discrete=True,
        shrink=0.8,
        stat="percent",
        element="bars",
        ax=axs[1, 1],
    )
    axs[1, 1].set_yscale("log")
    axs[1, 1].yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda y, _: "{:.16g}".format(y))
    )
    axs[1, 1].set_title(f"Rejection histogram, total : {output.rejected.sum()}")

    plt.tight_layout()
    plt.show()

    print("number of error bound : ", output.error_bound.sum())
