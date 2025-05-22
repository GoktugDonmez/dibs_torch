import jax
import jax.numpy as jnp
from jax.numpy import index_exp as index
from jax.scipy.special import logsumexp
from models.graph import topological_sort_jit


def stable_mean(fxs):
    # assumes fs are only positive
    jitter = 1e-30

    # Taking n separately we need non-zero
    stable_mean_psve_only = lambda fs, n: jnp.exp(
        logsumexp(jnp.log(fs)) - jnp.log(n + jitter)
    )

    f_xs_psve = fxs * (fxs > 0.0)
    f_xs_ngve = -fxs * (fxs < 0.0)

    n_psve = jnp.sum((fxs > 0.0))
    n_ngve = fxs.size - n_psve

    avg_psve = stable_mean_psve_only(f_xs_psve, n_psve)
    avg_ngve = stable_mean_psve_only(f_xs_ngve, n_ngve)
    return (n_psve / fxs.size) * avg_psve - (n_ngve / fxs.size) * avg_ngve


def log_stable_mean_from_logs(fsx):
    lse = logsumexp(fsx)
    return lse - fsx.size


def acyclic_constr(g_mat, d):
    # NOTE this code is a copy-paste from the dibs implementation

    alpha = 1.0 / d
    M = jnp.eye(d) + alpha * g_mat

    M_mult = jnp.linalg.matrix_power(M, d)
    h = jnp.trace(M_mult) - d

    return h


def sample_y(key, true_gmat, edges, rho):
    # Assumes Bernoulli expert
    gmat_value = lambda e: true_gmat[e[0], e[1]]
    gmat_values = jax.vmap(gmat_value)(edges)

    bernoulli_p = lambda p: jax.lax.cond(
        p == 1,
        lambda _: rho,
        lambda _: 1 - rho,
        None,
    ).astype(jnp.float32)
    bernoulli_ps = jax.vmap(bernoulli_p)(gmat_values)

    y = jax.random.bernoulli(key, bernoulli_ps)
    return y.astype(int)

# Define a layer forward pass with activation
def layer_with_activation(x, weights, bias):
    """Forward pass for a hidden layer with tanh activation"""
    return jax.nn.relu(weights @ x + bias)

# Define a linear forward pass (no activation)
def linear_layer(x, weights, bias):
    """Forward pass for the output layer (linear, no activation)"""
    return weights @ x + bias

# Forward function with separate handling for the output layer
def forward(x, weights_list, bias_list, d):
    """
    Forward pass through the network with a scalar input x
    The final layer is linear (no activation function)
    """
    # Reshape scalar to column vector
    activation = jnp.reshape(x, (d, 1))

    # Apply hidden layers with tanh activation
    for i in range(len(weights_list) - 1):
        activation = layer_with_activation(activation, weights_list[i], bias_list[i])

    # Apply final layer without activation (linear output)
    activation = linear_layer(activation, weights_list[-1], bias_list[-1])

    # Return scalar output
    return activation.squeeze()[()]  # Convert to scalar

# Pytree version
# @jax.jit
def forward_pytree(x, params_list, d):
    """Forward pass that takes a pytree of parameters"""
    # Extract weights and biases
    weights = [p['weights'] for p in params_list]
    biases = [p['bias'] for p in params_list]
    return forward(x, weights, biases, d)


def resnet_layer(x, weights, bias):
    """Residual layer: relu(Wx+b) + x"""
    # Ensure x and Wx+b match in shape
    h = jax.nn.relu(weights @ x + bias)
    return h + x

def forward_resnet(x, weights_list, bias_list, d):
    """ResNet forward pass for scalar input x"""
    activation = jnp.reshape(x, (d, 1))

    for i in range(len(weights_list) - 1):
        activation = resnet_layer(activation, weights_list[i], bias_list[i])
        # Optionally, add linear projection if shapes change:
        # if activation.shape != h.shape:
        #   activation = linear_projection(activation)

    # Final layer (no residual connection)
    activation = linear_layer(activation, weights_list[-1], bias_list[-1])
    return activation.squeeze()[()]

def forward_resnet_pytree(x, params_list, d):
    weights = [p['weights'] for p in params_list]
    biases = [p['bias'] for p in params_list]
    return forward_resnet(x, weights, biases, d)

def sample_x(key, hard_gmat, theta, n_samples, hparams, intervs=None, iscm=False):
    # Aiming to be agnostic to linear and nonlinear theta
    # theta represents the weights of a neural network
    # (W_0, b_0), ..., (W_l, b_l) W_i :: dxd, b_i :: dx1
    d = hard_gmat.shape[0]
    noise_std = hparams["noise_std"]
    parent_means = hparams["parent_means"]
    assert noise_std.size == parent_means.size == hard_gmat.shape[-1]

    # NOTE Will go to infinite loop if gmat not a DAG
    toporder = topological_sort_jit(hard_gmat)  # hparams["toporder"]

    if iscm:
        # If iSCM, the functional mechanisms are updated in ancestral sampling
        # TODO Maybe use dict.get ?
        assert "pop_means" in hparams.keys() and "pop_stds" in hparams.keys()
        pop_means = hparams["pop_means"]
        pop_stds = hparams["pop_means"]
    else:
        pop_means = jnp.zeros((d,))
        pop_stds = jnp.ones((d,))

    intervs = -1 * jnp.ones((1, 2)) if intervs is None else intervs
    assert intervs.size == 2
    if not isinstance(theta, dict):
    # if theta.shape == hard_gmat.shape:
        # xs because technically this generates output of shape (n_samples, )
        scm_forward = lambda xs, i: xs @ (hard_gmat * theta)[:, i]
    else:
        branches = [lambda: theta[i] for i in range(d)]
        secure_lookup = lambda index: jax.lax.switch(index, branches)
        scm_forward = lambda xs, i: jax.vmap(lambda x: forward_resnet_pytree(x * hard_gmat[:, i], secure_lookup(i), d))(xs)

    def sampler_i(
        i,
        key,
        hard_gmat,
        theta,
        x,
    ):
        # assumes atomic interventions
        current_var = toporder[i]
        i_var, i_val = intervs.reshape(-1)[0], intervs.reshape(-1)[1]
        parent_mask = hard_gmat[:, current_var] == 1.0

        # evaluate SCM function
        key, _subk = jax.random.split(key)
        exog_noise = noise_std[current_var] * jax.random.normal(key, shape=(n_samples,))
        x = jax.lax.cond(
            sum(parent_mask) == 0.0,  # if root node
            lambda _: x.at[index[:, current_var]].set(
                (parent_means[current_var] + exog_noise - pop_means[current_var])/pop_stds[current_var]
            ),  # no parents: just noise
            lambda _: x.at[index[:, current_var]].set(
                # TODO can optimize the line below by not computing the full matrix product, only the column we actually need
                (scm_forward(x, current_var) + exog_noise - pop_means[current_var])/pop_stds[current_var]
            ),
            operand=None,
        )

        # if we intervene, overwrite the SCM-computed value with the intervention value:
        # This can be done here since the sampler_var is called in topological order
        x = jax.lax.cond(
            jnp.any(current_var == i_var), # current_var cannot be negative, so if intervs is None, this condition is never satisfied
            lambda _: x.at[index[:, current_var]].set(i_val),
            lambda _: x,
            operand=None,
        )
        return (
            key,
            hard_gmat,
            theta,
            x,
        )

    d = hard_gmat.shape[-1]
    x = jnp.zeros((n_samples, d))

    key, hard_gmat, theta, x = jax.lax.fori_loop(0, d,
                                                 lambda i, args: sampler_i(i, *args), (key, hard_gmat, theta, x,),)
    return x

def sample_x_old(key, hard_gmat, theta, n_samples, hparams, intervs=None):
    # Given single linear model, sample observations
    # Assumes samples from the SCM model is provided
    # Atomic interventions only
    noise_std = hparams["noise_std"]
    parent_means = hparams["parent_means"]
    assert noise_std.size == parent_means.size == hard_gmat.shape[-1]

    # NOTE Will go to infinite loop if gmat not a DAG
    toporder = topological_sort_jit(hard_gmat)  # hparams["toporder"]

    intervs = -1 * jnp.ones((1, 2)) if intervs is None else intervs
    assert intervs.size == 2
    if theta.shape == hard_gmat.shape:
        scm_forward = lambda x: x @ (hard_gmat * theta)
    else:
        scm_forward = lambda x: None

    def sampler_i(
        i,
        key,
        hard_gmat,
        theta,
        x,
    ):
        # assumes atomic interventions
        current_var = toporder[i]
        i_var, i_val = intervs.reshape(-1)[0], intervs.reshape(-1)[1]
        parent_mask = hard_gmat[:, current_var] == 1.0

        # evaluate SCM function
        key, _subk = jax.random.split(key)
        exog_noise = noise_std[current_var] * jax.random.normal(key, shape=(n_samples,))
        x = jax.lax.cond(
            sum(parent_mask) == 0.0,  # if root node
            lambda _: x.at[index[:, current_var]].set(
                parent_means[current_var] + exog_noise
            ),  # no parents: just noise
            lambda _: x.at[index[:, current_var]].set(
                # TODO can optimize the line below by not computing the full matrix product, only the column we actually need
                (x @ (hard_gmat * theta))[:, current_var] + exog_noise  # linear model
            ),
            operand=None,
        )

        # if we intervene, overwrite the SCM-computed value with the intervention value:
        # This can be done here since the sampler_var is called in topological order
        x = jax.lax.cond(
            jnp.any(current_var == i_var), # current_var cannot be negative, so if intervs is None, this condition is never satisfied
            lambda _: x.at[index[:, current_var]].set(i_val),
            lambda _: x,
            operand=None,
        )
        return (
            key,
            hard_gmat,
            theta,
            x,
        )

    d = hard_gmat.shape[-1]
    x = jnp.zeros((n_samples, d))

    key, hard_gmat, theta, x = jax.lax.fori_loop(0, d,
                                                 lambda i, args: sampler_i(i, *args), (key, hard_gmat, theta, x,),)
    return x


def mean_f_over_ivals(key, i, hard_gmat_particles, theta_particles, hparams):
    # Take a lot of particles, and return the highest mean f value based on interventions
    n_particles = hard_gmat_particles.shape[0]
    if isinstance(theta_particles, jnp.ndarray):
        mean_fs_per_ivals = jnp.mean(
            jax.vmap(
                lambda p: f_distr_per_intervention_i(
                    key,
                    i,
                    hard_gmat_particles[p, ...],
                    theta_particles[p, ...],
                    hparams,
                )
            )(jnp.arange(n_particles)),
            axis=(0, 2),
        )
    else:
        mean_fs_per_ivals = jnp.mean(
            jax.vmap(
                lambda hard_gmat, theta: f_distr_per_intervention_i(
                    key,
                    i,
                    hard_gmat,
                    theta,
                    hparams,
                ), in_axes=(0, 0)
            )(hard_gmat_particles, theta_particles),
            axis=(0, 2),
        )

    return mean_fs_per_ivals


def interv_distr(key, hard_gmat_particles, theta_particles, hparams):
    # Get optimal interventional distribution by
    # sampling over all particles

    # TODO Allow for repeatedly sampling graphs from z
    # return 1.0

    (
        n_particles,
        _d,
        _d,
    ) = hard_gmat_particles.shape

    # TODO Allow for more resampling of hard graphs from z particles if necessary

    if isinstance(theta_particles, jnp.ndarray):
        max_interv_distr = jax.vmap(
            lambda p: best_intervention_distr(
                key, hard_gmat_particles[p, ...], theta_particles[p, ...], hparams
            )
        )(jnp.arange(n_particles))
    else:
        max_interv_distr = jax.vmap(lambda hard_gmat, theta: best_intervention_distr(key, hard_gmat, theta, hparams),
                                    in_axes=(0, 0))(hard_gmat_particles, theta_particles)
    # variable-value samples of best intervention
    return max_interv_distr.reshape(-1, 2)


def f_distr_per_intervention_i(key, i, hard_gmat, theta, hparams):
    # Perform intervention on variable i across a grid of interv values
    # For each do(i=ival) generate hparams["n_scm_samples"] samples of f(x_target)
    fx_do_i_ival = lambda ival: jax.vmap(hparams["f"])(
        sample_x(
            key,
            hard_gmat,
            theta,
            hparams["n_scm_samples"],
            hparams,
            jnp.array([i, ival]),
        )[:, -1]
    )
    return jax.vmap(fx_do_i_ival)(
        jnp.linspace(hparams["xmin"], hparams["xmax"], hparams["interv_discretization"])
    )


def best_intervention(key, hard_gmat, theta, hparams):
    # Given a model, return the best intervention which
    # maximizes some function f, based on the mean of f

    d = hard_gmat.shape[-1]
    interv_domain = jnp.linspace(
        hparams["xmin"], hparams["xmax"], hparams["interv_discretization"]
    )

    key, _subk = jax.random.split(key)
    fs_do_all = jax.vmap(
        lambda i: f_distr_per_intervention_i(key, i, hard_gmat, theta, hparams)
    )(jnp.arange(d - 1))

    # [d - 1 x interv_discretization x 1] # Mean of f samples at each intervention value
    avg_fs = jnp.mean(fs_do_all, axis=-1)

    linear_index = jnp.argmax(avg_fs)
    i, val_index = jnp.unravel_index(linear_index, avg_fs.shape)
    return jnp.array([i, interv_domain[val_index]])


def best_intervention_distr(key, hard_gmat, theta, hparams):
    # Return distribution of best interventions given a specific model

    hparams_single_scm_sample = hparams.copy()
    hparams_single_scm_sample.update({"n_scm_samples": 1})

    key, _subk = jax.random.split(key)
    keys = jax.random.split(
        key, hparams["n_scm_samples"]
    )  # nongrad MC samples for h(G) and denominator computation
    return jax.vmap(
        lambda k: best_intervention(k, hard_gmat, theta, hparams_single_scm_sample)
    )(keys)


def process_dag(gmat):
    # Return empty graph if the graph has cycles
    # Also return epsilon noise variances in these cases since
    # TODO Maybe use weighted particles and put weight as 0 if the graph is cyclic?

    n_vars = gmat.shape[-1]
    h_G = acyclic_constr(gmat, n_vars)
    is_cyclic = (h_G > 0).astype(int)  # if 1, thats bad
    target_has_no_parents = (jnp.sum(gmat[:, -1]) == 0).astype(int)  # if 1, thats bad.
    zero_out = jax.lax.bitwise_or(is_cyclic, target_has_no_parents)

    gmat_factor = jax.lax.cond(
        jnp.allclose(zero_out, 1.0), lambda _: 0.0, lambda _: 1.0, None
    )
    # sigmas_ = jax.lax.cond(
    #     jnp.allclose(zero_out, 1.0),
    #     lambda _: 1e-10 * jnp.ones_like(sigmas),
    #     lambda _: sigmas,
    #     None,
    # )
    return gmat_factor * gmat  # , sigmas_
