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


def fit_sghmc(key, data, grad_logdensity_fn, hparams, d=None):
    # TODO Put model and inference hyperparameters separately in the hparams dict
    # TODO Get MCMC metrics like what Stan gives, e.g. accepted steps
    # TODO Make the grad logdensity be a function of the model only
    d = data["x"].shape[-1]
    buffer_indices = posterior_buffer_indices(
        hparams["steps"], hparams["burn_in"], hparams["n_samples"]
    )

    def sghmc_single_chain(key):
        key, _subkey = jax.random.split(key)
        sghmc = blackjax.sgmcmc.sghmc.as_top_level_api(
            grad_estimator=lambda p, x: grad_logdensity_fn(
                key, x, p
            ),  # alpha=0.9, beta=0.99
        )

        key, _subkey = jax.random.split(key)
        sghmc_state = initialize_params(key, d, d)
        zs, thetas = [], []

        sghmc_update = jax.jit(sghmc.step)
        for t in tqdm(range(hparams["steps"]), file=sys.stdout):
            # print(sghmc_state["theta"])
            key, _subkey = jax.random.split(key)
            sghmc_state["t"] = sghmc_state["t"] + 1
            sghmc_state = sghmc_update(key, sghmc_state, data, hparams["lr"])

            if t in buffer_indices:
                zs.append(sghmc_state["z"])
                thetas.append(sghmc_state["theta"])
        z_samples = jnp.array(zs)
        hard_gmat_samples = hard_gmat_particles_from_z(z_samples)
        model = {
            "zs": z_samples,
            "hard_gmats": hard_gmat_samples,
            "thetas": jnp.array(thetas),
        }
        return model

    keys = jax.random.split(key, hparams["mcmc_chains"])
    all_chains = jax.vmap(sghmc_single_chain)(keys)
    model = {}
    for key, value in all_chains.items():
        model[key] = value.reshape(-1, *value.shape[2:])

    return model
