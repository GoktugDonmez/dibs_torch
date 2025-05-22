import blackjax
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from models.dibs import hard_gmat_particles_from_z


def initialize_params(key, d, k):
    z_std = 1.0 / jnp.sqrt(k)
    key, _subk = jax.random.split(key)
    z = jax.random.normal(key, shape=(d, k, 2)) * z_std
    key, _subk = jax.random.split(key)
    theta = jax.random.normal(key, shape=(d, d))
    t = jnp.ones((1,)) + 1

    return {"z": z, "theta": theta, "t": t}


def fit_nuts(key, data, logdensity_fn, hparams, print_freq=50, d=None):
    # TODO Put model and inference hyperparameters separately in the hparams dict
    # TODO Get MCMC metrics like what Stan gives, e.g. accepted steps
    d = data["x"].shape[-1]
    # Make the grad logdensity be a function of the model only
    key, _subkey = jax.random.split(key)
    initial_position = initialize_params(key, d, d)

    # TODO Clean this later, put coeffs dict below as part of hparams
    inverse_mass_matrix_coeffs = {
        "z": 2.0 / initial_position["z"].size,
        "theta": 5.0 / initial_position["theta"].size,
        "t": 1.0 / initial_position["t"].size,
    }
    inverse_mass_matrix = ravel_pytree(
        {
            key: inverse_mass_matrix_coeffs[key] * jnp.ones_like(initial_position[key])
            for key in initial_position.keys()
        }
    )[0]

    key, _subkey = jax.random.split(key)
    nuts = blackjax.mcmc.nuts.as_top_level_api(
        logdensity_fn=lambda p: logdensity_fn(key, data, p),
        step_size=hparams["lr"],
        inverse_mass_matrix=inverse_mass_matrix,
    )
    nuts_state = nuts.init(position=initial_position)

    nuts_update = jax.jit(nuts.step)
    for t in range(hparams["steps"]):
        nuts_state.position["t"] = nuts_state.position["t"] + 1

        key, _subkey = jax.random.split(key)
        nuts_state, nuts_info = nuts_update(key, nuts_state)
        if t % print_freq == 0:
            print(
                f"{nuts_info.is_turning=}, {nuts_info.is_divergent=} {nuts_info.num_integration_steps=}"
            )

    z_particles = nuts_state.position["z"]
    hard_gmat_particles = hard_gmat_particles_from_z(z_particles)
    model = {
        "zs": z_particles,
        "hard_gmats": hard_gmat_particles,
        "thetas": nuts_state.position["theta"],
    }
    return model
