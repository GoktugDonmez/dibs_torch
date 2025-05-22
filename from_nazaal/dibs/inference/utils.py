def posterior_buffer_indices(total_steps, burn_in, n_samples):
    post_burn_in_indices = [i for i in range(int(total_steps * burn_in), total_steps)]
    # If only one sample requested, return the last index
    if n_samples == 1:
        return [post_burn_in_indices[-1]]
    # Calculate step size using n_samples - 1 to ensure we hit the last index
    step = (len(post_burn_in_indices) - 1) / (n_samples - 1)
    # Use float multiplication and round after to avoid accumulated rounding errors
    indices = [
        post_burn_in_indices[min(round(i * step), len(post_burn_in_indices) - 1)]
        for i in range(n_samples)
    ]
    return indices
