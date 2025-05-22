import sys

import blackjax
import jax
import jax.numpy as jnp
from inference.utils import posterior_buffer_indices
from models.dibs import hard_gmat_particles_from_z
from tqdm import tqdm


def initialize_params(key, d, k):
    z_std = 1.0 / jnp.sqrt(k)
    key, _subk = jax.random.split(key)
    z = jax.random.normal(key, shape=(d, k, 2)) * z_std
    key, _subk = jax.random.split(key)
    theta = jax.random.normal(key, shape=(d, d))
    t = jnp.ones((1,))

    return {"z": z, "theta": theta, "t": t}


def fit_sgld(key, data, grad_logdensity_fn, hparams, d=None):
    # TODO Put model and inference hyperparameters separately in the hparams dict
    # TODO Get MCMC metrics like what Stan gives, e.g. accepted steps
    d = data["x"].shape[-1]
    # Make the grad logdensity be a function of the model only
    key, _subkey = jax.random.split(key)
    sgld = blackjax.sgmcmc.sgld.as_top_level_api(
        grad_estimator=lambda p, x: grad_logdensity_fn(key, x, p),
    )

    key, _subkey = jax.random.split(key)
    sgld_state = sgld.init(position=initialize_params(key, d, d))
    buffer_indices = posterior_buffer_indices(
        hparams["steps"], hparams["burn_in"], hparams["n_samples"]
    )
    zs, thetas = [], []

    svgd_update = jax.jit(sgld.step)
    for t in tqdm(range(hparams["steps"]), file=sys.stdout):
        key, _subkey = jax.random.split(key)
        sgld_state = svgd_update(
            key, sgld_state, minibatch=data, step_size=hparams["lr"]
        )
        sgld_state["t"] = (
            t
            * jnp.ones(
                (1,)
            )  # Dummy parameter used to update time-dependent hyperparams
        ) + 1

        if t in buffer_indices:
            zs.append(sgld_state["z"])
            thetas.append(sgld_state["theta"])

    z_samples = jnp.array(zs)
    hard_gmat_samples = hard_gmat_particles_from_z(z_samples)
    model = {
        "zs": z_samples,
        "hard_gmats": hard_gmat_samples,
        "thetas": jnp.array(thetas),
    }
    return model
