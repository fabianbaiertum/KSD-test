def bootstrap_ksd_statistics(
    rng: KeyArray,
    ys: Array,
    *,              #everything after this must be passed by keyword, not by position
    B: int,
    estimator: KSDAnalyticEstimator,
    theta_hat: Array,
    kernel: Kernel,
    null_class: type[UnnormalizedDist],
    grad_h_fn_hat: Callable[[Array, Array], Array],
) -> Array:
    """
    Draw B bootstrap resamples of size n (with replacement),
    re-estimate θ on each resample, and compute the KSD U-statistic.

    Returns
    -------
    T_vals : jnp.ndarray (B,)
        Vector of bootstrap statistics T^{(b)}.
    """
    n = ys.shape[0]

    def one_bootstrap(key: KeyArray) -> Array:
        # 1. resample indices with replacement
        idx = random.choice(key, n, shape=(n,), replace=True)
        ys_star = ys[idx]

        # 2. re-estimate θ on the resample
        theta_star = estimator(key, ys_star)

        h_fn_star = h_fn(theta_star, key)
        grad_h_fn_star=grad_h_fn(theta_star, x, xp)  ###TODO fix that x, xp are two data points
        # 3. compute T^{(b)} = KSD_q^2( p_{θ*} )
        T_b = tilde_ksd(X=samples,X_star=ys_star,n=n,theta_hat=theta_hat,theta_star=theta_star,h_fn_star=h_fn_star,grad_h_fn_star=grad_h_fn_star,grad_h_fn_hat=grad_h_fn_hat)
        return T_b                       # <-- set to 1.0 if you just want a placeholder

    # Split RNG into B independent keys and vmap over them
    keys = random.split(rng, B)
    T_vals = vmap(one_bootstrap)(keys)   # shape (B,)

    return T_vals

