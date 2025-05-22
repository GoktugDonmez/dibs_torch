from functools import partial

import jax
import jax.numpy as jnp
from blackjax.util import ravel_pytree
from jax.nn import sigmoid
from jax.scipy.special import logsumexp
from jax.scipy.stats import norm as jax_normal
from jax.tree_util import tree_map
from models.utils import acyclic_constr, stable_mean
from omegaconf import OmegaConf


def log_gaussian_likelihood(x, pred_mean, sigma=0.1):
    # x         :: [1 x d] | [n x d]
    # pred_mean :: [1 x d] | [n x d]
    # sigma     :: [1 x d] | float
    # assumes each sample is iid
    # note sigma is important! values relatively small to actual data can lead to theta exploding
    # todo learn sigma (i.e. exogenous noise variance) using mcmc

    if not isinstance(sigma, float):
        assert isinstance(sigma, jnp.ndarray)
        sigma = sigma.reshape(-1)
        assert sigma.shape[0] == pred_mean.shape[-1]
    else:
        assert isinstance(sigma, float)

    return jnp.sum(jax_normal.logpdf(x=x, loc=pred_mean, scale=sigma)).astype(float)


def log_bernoulli_likelihood(y, soft_gmat, rho) -> float:
    # Single sample y
    # NOTE Not a good idea to pass the whole graph adjacency matrix here
    inv_temperature = 1.0
    jitter = 1e-5  # Inference is quite sensitive to this jitter term
    i, j, y = y[0], y[1], y[2]
    g_ij = soft_gmat[i, j]
    p_tilde = rho + g_ij - 2 * rho * g_ij
    loglik = inv_temperature * (
        y * jnp.log(1 - p_tilde + jitter) + (1 - y) * jnp.log(p_tilde + jitter)
    )
    return loglik.astype(float)


def gumbel_soft_gmat(key, z, d, hparams):
    # Given single z, return soft adjacency matrix using 1 Gumbel sample
    # Generate adjacency matrix using Gumebl samples for reparametrized gradients
    cycle_mask = 1.0 - jnp.eye(d)
    nodes = jnp.arange(d)

    gumbel_soft_gmat_ij = lambda _l, _z, i, j: (
        sigmoid(
            (_l[i, j] + (hparams["alpha"] * jnp.dot(_z[i, :, 0], _z[j, :, 1])))
            * hparams["tau"]
        )
    )

    l = lambda k: jax.random.logistic(k, shape=(d, d))

    # Apply gumbel_soft_gmat_ij to each [i,j] entry in graph adajacency matrix in paralllel
    return (
        cycle_mask
        * jax.vmap(
            lambda i: jax.vmap(lambda j: gumbel_soft_gmat_ij(l(key), z, i, j))(nodes)
        )(nodes).T
    )


def scores(z, d, hparams):
    cycle_mask = 1.0 - jnp.eye(d)
    nodes = jnp.arange(d)
    scores_ij = lambda _z, i, j: hparams["alpha"] * jnp.dot(_z[i, :, 0], _z[j, :, 1])
    return (
        cycle_mask
        * jax.vmap(lambda i: jax.vmap(lambda j: scores_ij(z, i, j))(nodes))(nodes).T
    )


def bernoulli_soft_gmat(z, d, hparams):
    # Given a single z particle, return soft adjacency matrix using Bernoulli samples
    cycle_mask = 1.0 - jnp.eye(d)
    nodes = jnp.arange(d)
    soft_gmat_ij = lambda _z, i, j: sigmoid(
        hparams["alpha"] * jnp.dot(_z[i, :, 0], _z[j, :, 1])
    )
    # apply soft_gmat_ij to each [i,j] entry in graph adjacency matrix in paralllel
    return (
        cycle_mask
        * jax.vmap(lambda i: jax.vmap(lambda j: soft_gmat_ij(z, i, j))(nodes))(nodes).T
    )


def log_full_likelihood(data, soft_gmat, hard_gmat, theta, hparams):
    # NOTE Assumes observational noise is known
    log_full_obs_likelihood = log_gaussian_likelihood(
        data["x"], data["x"] @ (theta * soft_gmat), sigma=0.1
    )

    log_full_expert_likelihood, inv_temperature = 0.0, 0.0
    # TODO Make it give error is y is out of bounds
    if data.get("y", None) is not None:
        inv_temperature = hparams[
            "temp_ratio"
        ]  # likelihoods are weighed only if expert data is available
        log_full_expert_likelihood = jnp.sum(
            jax.vmap(lambda y: log_bernoulli_likelihood(y, soft_gmat, hparams["rho"]))(
                data["y"]
            )
        )

    return inv_temperature * log_full_expert_likelihood + log_full_obs_likelihood


def acyclic_constr_mc(key, z, d, hparams):
    key, _subkey = jax.random.split(key)
    keys = jax.random.split(
        key, hparams["n_nongrad_mc_samples"]
    )  # nongrad MC samples for h(G) and denominator computation
    acyclic_constr_mc_samples = jax.vmap(
        lambda k: acyclic_constr(
            jax.random.bernoulli(k, bernoulli_soft_gmat(z, d, hparams)), d
        )
    )(keys)
    return stable_mean(acyclic_constr_mc_samples)


def gumbel_acyclic_constr_mc_stable_mean(key, z, d, hparams):
    key, _subkey = jax.random.split(key)
    soft_gmat = gumbel_soft_gmat(key, z, d, hparams)

    keys = jax.random.split(
        key, hparams["n_nongrad_mc_samples"]
    )  # nongrad mc samples for h(g) and denominator computation
    gumbel_acyclic_constr_mc_samples = jax.vmap(
        lambda k: acyclic_constr(jax.random.bernoulli(k, soft_gmat), d)
    )(keys)
    return stable_mean(gumbel_acyclic_constr_mc_samples)


def gumbel_grad_acyclic_constr(key, z, d, hparams):
    # NOTE Important to use soft graph here, else inner product of latent variables behave weirdly
    key, _subkey = jax.random.split(key)
    soft_gmat = lambda _z: gumbel_soft_gmat(key, _z, d, hparams)
    key, _subkey = jax.random.split(key)
    # hard_gmat = lambda _z: jax.random.bernoulli(key, soft_gmat(_z))
    return jax.grad(lambda _z: acyclic_constr(soft_gmat(_z), d))(z)


def gumbel_grad_acyclic_constr_mc(key, z, d, hparams):
    # TODO Bundle into single function without using gumbel_grad_acyclic_constr
    key, _subkey = jax.random.split(key)
    keys = jax.random.split(
        key, hparams["n_nongrad_mc_samples"]
    )  # nongrad mc samples for h(g) and denominator computation
    gumbel_grad_acyclic_constr_mc_samples = jax.vmap(
        lambda k: gumbel_grad_acyclic_constr(k, z, d, hparams)
    )(keys)
    return jnp.mean(gumbel_grad_acyclic_constr_mc_samples, axis=0)


def gumbel_acyclic_constr_mc(key, z, d, hparams):
    key, _subkey = jax.random.split(key)
    keys = jax.random.split(
        key, hparams["n_nongrad_mc_samples"]
    )  # nongrad mc samples for h(g) and denominator computation
    soft_gmat = lambda k: gumbel_soft_gmat(k, z, d, hparams)
    h_samples = jax.vmap(lambda k: acyclic_constr(soft_gmat(k), d))(keys)
    return jnp.mean(h_samples)


def grad_z_log_joint_gumbel(key, data, opt_params, nonopt_params):
    # equation 12, to be used
    # gradient of the log density with respect to z

    # NOTE mean in gaussian likelihood below can be more general e.g. a nn
    # NOTE Assumes no element in theta is exactly zero, otherwise some NaN problems arise

    beta = nonopt_params["beta"]
    d = opt_params["z"].shape[0]

    # Numerator in Equation 10
    key, _subkey = jax.random.split(key)
    grad_log_z_prior_mc = (
        -beta * gumbel_grad_acyclic_constr_mc(key, opt_params["z"], d, nonopt_params)
        - (1 / nonopt_params["sigma_z"] ** 2) * opt_params["z"]
    )

    # TODO Check if z_likelihood is the same as original implementation
    # log of density in numerator and denominator of Equation A.28, as a function of z
    log_density_z = (
        lambda k, z: log_full_likelihood(
            data=data,
            soft_gmat=gumbel_soft_gmat(k, z, d, nonopt_params),
            hard_gmat=None,  # jax.random.bernoulli(k, gumbel_soft_gmat(k, z, d, nonopt_params)),
            theta=nonopt_params["theta"],
            hparams=nonopt_params,
        )
        + log_theta_prior(
            nonopt_params["theta"]
            * gumbel_soft_gmat(
                k, z, d, nonopt_params
            ),  # * hard_gmat_particles_from_z(z),
            params={"theta_prior_mean": jnp.zeros_like(nonopt_params["theta"])},
            hparams={"theta_prior_sigma": 1.0},
        )
    )

    key, _subkey = jax.random.split(key)
    keys = jax.random.split(key, nonopt_params["n_grad_mc_samples"])
    z_likelihood_samples_numerator = jax.vmap(
        lambda k: log_density_z(k, opt_params["z"])
    )(keys)
    z_grad_samples = jax.vmap(
        lambda k: jax.grad(lambda z: log_density_z(k, z))(opt_params["z"])
    )(keys)

    lse_numerator = tree_map(
        lambda leaf_z: logsumexp(
            a=expand_by(z_likelihood_samples_numerator, leaf_z.ndim - 1),
            b=leaf_z,
            axis=0,
            return_sign=True,
        )[0],
        z_grad_samples,
    )
    sign_lse_numerator = tree_map(
        lambda leaf_z: logsumexp(
            a=expand_by(z_likelihood_samples_numerator, leaf_z.ndim - 1),
            b=leaf_z,
            axis=0,
            return_sign=True,
        )[1],
        z_grad_samples,
    )

    # NOTE Using same samples reduces variance, otherwise results are numerically unstable
    z_likelihood_samples_denominator = z_likelihood_samples_numerator
    lse_denominator = logsumexp(a=z_likelihood_samples_denominator, axis=0)

    stable_grad = grad_log_z_prior_mc + tree_map(
        lambda sign_leaf_z, log_leaf_z: (
            sign_leaf_z
            * jnp.exp(
                log_leaf_z
                - jnp.log(nonopt_params["n_grad_mc_samples"])  # Technically cancels out
                - lse_denominator
                + jnp.log(nonopt_params["n_grad_mc_samples"])
            )
        ),
        sign_lse_numerator,
        lse_numerator,
    )
    return stable_grad


def expand_by(arr, n):
    # From DiBS code
    return jnp.expand_dims(arr, axis=tuple(arr.ndim + j for j in range(n)))


def log_theta_prior(theta, params, hparams, zero=False):
    if zero:
        return 0.0
    return log_gaussian_likelihood(
        theta, params["theta_prior_mean"], hparams["theta_prior_sigma"]
    )


def hard_gmat_particles_from_z(z_particles):
    single_z = len(z_particles.shape) == 3
    if single_z:
        d, _k, _ = z_particles.shape
        return (scores(z_particles, d, {"alpha": 1.0}) > 0.0).astype(float)

    n_particles, d, _k, _ = z_particles.shape
    hard_gmat_particles = jax.vmap(
        lambda p: (scores(z_particles[p], d, {"alpha": 1.0}) > 0.0).astype(float)
    )(jnp.arange(n_particles))
    return hard_gmat_particles


@partial(jax.custom_jvp, nondiff_argnums=(0, 1, 3))
def log_joint(key, data, params, hparams):
    # Given one sample of the parameters, return the log joint density value
    t = params["t"].reshape(-1)[0].astype(int)
    hparams = update_dibs_hparams(hparams, t)
    key = jax.random.fold_in(key, t)

    d = data["x"].shape[1]
    log_likelihood = log_full_likelihood(
        data=data,
        soft_gmat=bernoulli_soft_gmat(params["z"], d, hparams),
        hard_gmat=None,  # hard_gmat_particles_from_z(params["z"]),
        theta=params["theta"],
        hparams=hparams,
    )

    log_prior = (
        -hparams["beta"] * gumbel_acyclic_constr_mc(key, params["z"], d, hparams)
        + jnp.sum(
            jax_normal.logpdf(
                x=params["z"], loc=jnp.zeros_like(params["z"]), scale=hparams["sigma_z"]
            )
        ).astype(float)
    ) + log_theta_prior(
        params["theta"] * bernoulli_soft_gmat(params["z"], d, hparams),
        params={"theta_prior_mean": jnp.zeros_like(params["theta"])},
        hparams={"theta_prior_sigma": 1.0},
    )

    return log_likelihood + log_prior


# Needed for NUTS, HMC etc
# See https://blackjax-devs.github.io/blackjax/examples/howto_custom_gradients.html
@log_joint.defjvp
def log_joint_jvp(key, data, hparams, primals, tangents):
    (primals_in,) = primals
    (tangents_in,) = tangents

    primals_out = log_joint(key, data, primals_in, hparams)
    tangents_out = jnp.sum(
        ravel_pytree(
            tree_map(
                lambda x, y: x * y,
                grad_log_joint(key, data, primals_in, hparams),
                tangents_in,
            )
        )[0]
    )
    return primals_out, tangents_out


def grad_theta_log_joint(key, data, opt_params, nonopt_params):
    # gradient of the log density with respect to theta
    # NOTE mean in gaussian likelihood below can be more general e.g. a nn
    # NOTE assumes no prior over theta

    # TODO Make this work for theta which can in general be a PyTree

    d = data["x"].shape[1]

    # Numerator in Equation 11
    # log since this will be passed to stable mean which takes log of
    # what we want to sum
    # No need for gumbel samples here
    # log of density in numerator and denominator of Equation A.33, as a function of theta
    theta_log_joint = (
        lambda _k, theta: log_full_likelihood(
            data=data,
            soft_gmat=bernoulli_soft_gmat(nonopt_params["z"], d, nonopt_params),
            hard_gmat=None,  # jax.random.bernoulli(k, bernoulli_soft_gmat(nonopt_params["z"], d, nonopt_params)),
            theta=theta,
            hparams=nonopt_params,
        )
        + log_theta_prior(
            theta
            * bernoulli_soft_gmat(
                nonopt_params["z"], d, nonopt_params
            ),  # * hard_gmat_particles_from_z(nonopt_params["z"]),
            params={"theta_prior_mean": jnp.zeros_like(theta)},
            hparams={"theta_prior_sigma": 1.0},
        )
    )

    # random number generation keys
    key, _subkey = jax.random.split(key)
    keys = jax.random.split(key, nonopt_params["n_grad_mc_samples"])
    theta_log_joint_samples_numerator = jax.vmap(
        lambda k: theta_log_joint(k, opt_params["theta"])
    )(keys)
    theta_grad_samples = jax.vmap(
        lambda k: jax.grad(lambda t: theta_log_joint(k, t))(opt_params["theta"])
    )(keys)

    lse_numerator = tree_map(
        lambda leaf_theta: logsumexp(
            a=expand_by(theta_log_joint_samples_numerator, leaf_theta.ndim - 1),
            b=leaf_theta,
            axis=0,
            return_sign=True,
        )[0],
        theta_grad_samples,
    )
    sign_lse_numerator = tree_map(
        lambda leaf_theta: logsumexp(
            a=expand_by(theta_log_joint_samples_numerator, leaf_theta.ndim - 1),
            b=leaf_theta,
            axis=0,
            return_sign=True,
        )[1],
        theta_grad_samples,
    )

    # NOTE Using same samples reduces variance, otherwise results are numerically unstable
    theta_log_joint_samples_denominator = theta_log_joint_samples_numerator
    lse_denominator = logsumexp(a=theta_log_joint_samples_denominator, axis=0)

    stable_grad = tree_map(
        lambda sign_leaf_theta, log_leaf_theta: (
            sign_leaf_theta
            * jnp.exp(
                log_leaf_theta
                - jnp.log(nonopt_params["n_grad_mc_samples"])
                - lse_denominator
                + jnp.log(nonopt_params["n_grad_mc_samples"])
            )
        ),
        sign_lse_numerator,
        lse_numerator,
    )
    return stable_grad


def grad_log_joint(key, data, params, hparams):
    # Update any time-dependent parameters e.g. values which are annealed
    # The 'params' argument is for the parameters that are actually learnt
    t = params["t"].reshape(-1)[0].astype(int)
    hparams = update_dibs_hparams(hparams, t)
    key = jax.random.fold_in(key, t)

    # # anneal_stop_ratio < 0 means annealing not used
    # step_cycle_ratio = hparams["steps"] / hparams["anneal_cycles"]
    # angelo_fortuin_anneal = (jnp.mod(t, step_cycle_ratio) / step_cycle_ratio) ** (
    #     hparams["anneal_speed"]
    #     * ((t / hparams["steps"]) < hparams["anneal_stop_ratio"])
    # )
    angelo_fortuin_anneal = 1.0

    grad_z = angelo_fortuin_anneal * grad_z_log_joint_gumbel(
        key, data, opt_params={"z": params["z"]}, nonopt_params=params | hparams
    )
    grad_theta = angelo_fortuin_anneal * grad_theta_log_joint(
        key, data, opt_params={"theta": params["theta"]}, nonopt_params=params | hparams
    )

    return {"z": grad_z, "theta": grad_theta, "t": jnp.array([0.0])}


def update_dibs_hparams(hparams, t):
    # note these values become tracers, so the hparam dict used for
    # inference cannot be used later in utility computations
    # way to get around this is to update d.copy()
    if not isinstance(hparams, dict):
        # Conversion needed since _partial_ instantiation in hydra
        # makes hparams an OmegaConf.dictconf
        hparams = OmegaConf.to_container(hparams, resolve=True)

    updated_hparams = hparams.copy()
    updated_hparams["tau"] = hparams["tau"]  # hparams["tau"] * (t + 1 / t)
    updated_hparams["alpha"] = hparams["alpha"] * (t + 1 / t)
    updated_hparams["beta"] = hparams["beta"] * (t + 1 / t)
    return updated_hparams
