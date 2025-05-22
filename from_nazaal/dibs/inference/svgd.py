import sys

import blackjax
import jax
import jax.numpy as jnp
import optax
from blackjax.vi.svgd import SVGDState, median_heuristic
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_map
from models.dibs import hard_gmat_particles_from_z
from models.utils import process_dag
from tqdm import tqdm


def rbf_sum(x, y, length_scale, params=["z", "theta"]):
    # Avoid RBF kernel over the dummary t parameter used to update hparams
    x = {k: x[k] for k in params}
    y = {k: y[k] for k in params}
    length_scale = {k: length_scale[k] for k in params}
    return jnp.sum(
        ravel_pytree(tree_map(blackjax.vi.svgd.rbf_kernel, x, y, length_scale))[0]
    )


def update_median_heuristic_own(state, median=True):
    # Updated to make bandwidth for Z and Theta to be chosen independently

    position, kernel_parameters, opt_state = state
    if median:
        updated_kernel_parameters = {
            "length_scale": {
                str(key): median_heuristic(
                    {"length_scale": kernel_parameters["length_scale"][key]},
                    {str(key): position[str(key)]},
                )["length_scale"]
                for key in kernel_parameters["length_scale"].keys()
            }
        }
    else:
        updated_kernel_parameters = {
            "length_scale": {"z": 5.0, "theta": 500, "t": 1.0},
        }

    # need {"length_scale": {"z": {"length_scale": ...}, "theta": ..., "t": ...,}} as output from median heuristic
    return SVGDState(position, updated_kernel_parameters, opt_state)


def initialize_params(key, d, k, n_particles=1):
    z_std = 1.0 / jnp.sqrt(k)
    key, _subk = jax.random.split(key)
    z = jax.random.normal(key, shape=(n_particles, d, k, 2)) * z_std
    key, _subk = jax.random.split(key)
    theta = jax.random.normal(key, shape=(n_particles, d, d))
    t = jnp.ones((n_particles,))

    return {"z": z, "theta": theta, "t": t}


def fit_svgd(key, data, grad_logdensity_fn, hparams, d=None):
    # TODO Put model and inference hyperparameters separately in the hparams dict
    # TODO Make case to start from a specific initial point
    d = data["x"].shape[-1]
    # Make the grad logdensity be a function of the model only
    key, _subkey = jax.random.split(key)
    svgd = blackjax.vi.svgd.as_top_level_api(
        grad_logdensity_fn=lambda p: grad_logdensity_fn(key, data, p),
        optimizer=optax.rmsprop(learning_rate=hparams["lr"]),
        kernel=rbf_sum,
        update_kernel_parameters=lambda s: update_median_heuristic_own(
            s, hparams["median_heuristic"]
        ),
    )

    key, _subkey = jax.random.split(key)
    svgd_state = svgd.init(
        initial_position=initialize_params(key, d, d, hparams["n_particles"]),
        kernel_parameters={"length_scale": {"z": 5.0, "theta": 500.0, "t": 1}},
    )

    svgd_update = jax.jit(svgd.step)
    for t in tqdm(range(hparams["steps"]), file=sys.stdout):
        svgd_state.particles["t"] = (
            t
            * jnp.ones(
                (hparams["n_particles"],)
            )  # Dummy parameter used to update time-dependent hyperparams
        ) + 1

        svgd_state = svgd_update(svgd_state)

    z_particles = svgd_state.particles["z"]
    hard_gmat_particles = hard_gmat_particles_from_z(z_particles)
    model = {
        "zs": z_particles,
        "hard_gmats": jax.vmap(process_dag)(
            hard_gmat_particles
        ),  # Empty out any cyclic graphs
        "thetas": svgd_state.particles["theta"],
    }
    return model
