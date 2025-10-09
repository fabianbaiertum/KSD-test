# ============================================================
# Optimized KSD / MMD power study (no wild bootstrap anywhere)
# - Fast RBF KSD core (no per-pair AD)
# - Pre-indexed composite bootstrap for ~KSD
# - Robust JAX<->CuPy transfer: zero-copy when possible, safe fallback otherwise
# - Parametric bootstrap batching knob
# ============================================================

from __future__ import annotations
from functools import partial, wraps
from typing import Generic, Protocol, Type, TypeVar, Callable, Iterable, Optional, Literal, Tuple, Union, cast, ParamSpec
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from math import ceil
import math

# ---- JAX setup (float32 by default) ----
import jax
import jax.numpy as jnp
from jax import Array, grad, jacfwd, jacrev, jit, vmap, tree_util, random, lax
from jax.numpy import atleast_2d
from jax.scipy.stats import multivariate_normal, norm
from jax.random import PRNGKey

jax.config.update("jax_enable_x64", False)  # use float32 unless you need 64-bit

# plotting + utils
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm as _tqdm
import pandas as pd

# CuPy-side (BFM)
import cupy as cp
from scipy.stats import norm as scipy_norm

try:                                   # JAX < 0.4.24
    from jax.random import KeyArray
except (ImportError, AttributeError):  # JAX ≥ 0.4.24
    KeyArray = jax.Array               # type: ignore[assignment]

# ---------------------------
# Helpers: JAX <-> CuPy (robust)
# ---------------------------
_dlpack_warned = False

def _jax_device_platform(x_jax) -> Optional[str]:
    """Best-effort device platform fetch ('cpu' | 'gpu' | 'tpu' | None)."""
    try:
        dev = x_jax.device() if callable(getattr(x_jax, "device", None)) else x_jax.device
        return getattr(dev, "platform", None)
    except Exception:
        return None

def jax_to_cupy(x_jax: Array) -> "cp.ndarray":
    """
    Prefer zero-copy via DLPack if JAX array lives on GPU and CuPy can consume it.
    Otherwise, fall back to a safe host copy to avoid crashes.
    """
    global _dlpack_warned
    try:
        if _jax_device_platform(x_jax) == "gpu":
            # New DLPack API: pass producer array directly to CuPy
            return cp.from_dlpack(x_jax)
    except Exception as e:
        if not _dlpack_warned:
            print(f"[BFM] DLPack zero-copy unavailable ({e}); falling back to host copy.")
            _dlpack_warned = True

    # Safe cross-device copy (works for CPU JAX -> GPU CuPy, etc.)
    return cp.asarray(np.asarray(x_jax), dtype=np.float32)

# ==============
# Base protocols
# ==============
X = TypeVar("X")
P = TypeVar("P")

class Distribution(ABC, Generic[P]):
    @abstractmethod
    def get_params(self) -> P:
        ...

class SampleableDist(Distribution[P], ABC, Generic[P, X]):
    def sample(self, rng: KeyArray, n: int) -> X:
        return self.sample_with_params(rng, self.get_params(), n)

    @staticmethod
    @abstractmethod
    def sample_with_params(rng: KeyArray, params: P, n: int) -> X:
        ...

class UnnormalizedDist(Distribution[P], ABC, Generic[P]):
    def score(self, x: Array) -> Array:
        return self.score_with_params(self.get_params(), x)

    @staticmethod
    @abstractmethod
    def score_with_params(params: P, x: Array) -> Array:
        ...

    def unnorm_log_prob(self, x: Array) -> Array:
        return self.unnorm_log_prob_with_params(self.get_params(), x)

    @classmethod
    @abstractmethod
    def unnorm_log_prob_with_params(cls, params: P, x: Array) -> Array:
        ...

class NormalizedDist(UnnormalizedDist[P], ABC, Generic[P]):
    def log_prob(self, x: Array) -> Array:
        return self.log_prob_with_params(self.get_params(), x)

    @staticmethod
    @abstractmethod
    def log_prob_with_params(params: P, x: Array) -> Array:
        ...

    @classmethod
    def unnorm_log_prob_with_params(cls, params: P, x: Array) -> Array:
        return cls.log_prob_with_params(params, x)

class ExpFamilyDist(Protocol):
    @staticmethod
    @abstractmethod
    def natural_parameter(params: Array) -> Array:
        ...

    @staticmethod
    @abstractmethod
    def natural_parameter_inverse(eta_val: Array) -> Array:
        ...

    @staticmethod
    @abstractmethod
    def sufficient_statistic(x: Array) -> Array:
        ...

    @staticmethod
    @abstractmethod
    def b(x: Array) -> Array:
        ...

# =========
# Estimator
# =========
P_ = TypeVar("P_")

class Estimator(ABC, Generic[P_]):
    @abstractmethod
    def __call__(self, rng: KeyArray, ys: Array) -> P_:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...

@dataclass
class TrueEstimator(Estimator[P_]):
    theta: P_
    def __call__(self, rng: KeyArray, ys: Array) -> P_:
        return self.theta
    @property
    def name(self) -> str: return "true"

Scalar = Array

# =======
# Kernels
# =======
KernelLike = Callable[[Array, Array], Array]

class Kernel(ABC):
    @abstractmethod
    def __call__(self, x1: Array, x2: Array) -> Scalar:
        ...

@dataclass(frozen=True, eq=True)
class GaussianKernel(Kernel):
    l: float
    def __call__(self, x1: Array, x2: Array) -> Scalar:
        # exp(-||x1-x2||^2 / (2 l^2))
        return jnp.exp(-((x1 - x2) ** 2).sum() / (2.0 * (self.l ** 2)))

@dataclass(frozen=True, eq=True)
class IMQKernel(Kernel):
    l: float
    gamma: float = 0.5
    def __post_init__(self) -> None:
        assert self.l >= 0.0
        assert 0.0 <= self.gamma <= 1.0
    def __call__(self, x1: Array, x2: Array) -> Scalar:
        return (1.0 + ((x1 - x2) ** 2).sum() / (2.0 * self.l ** 2)) ** (-self.gamma)

class SumKernel(Kernel):
    def __init__(self, kernels: Iterable[Kernel]) -> None:
        self.kernels = tuple(kernels)
    def __call__(self, x1: Array, x2: Array) -> Array:
        return jnp.array([k(x1, x2) for k in self.kernels]).sum()
    def __hash__(self) -> int: return hash(self.kernels)
    def __eq__(self, o: object) -> bool:
        if not isinstance(o, SumKernel): return False
        if len(self.kernels) != len(o.kernels): return False
        return all(k1 == k2 for (k1, k2) in zip(self.kernels, o.kernels))

def gram(x1: Array, x2: Array, kernel: KernelLike) -> Array:
    assert x1.ndim == 2 and x2.ndim == 2
    return vmap(lambda x: vmap(lambda y: kernel(x, y))(x2))(x1)

# ==============
# JAX utilities
# ==============
T_co = TypeVar("T_co")

def tree_concatenate(trees: Iterable[T_co]) -> T_co:
    leaves, treedefs = zip(*[tree_util.tree_flatten(tree) for tree in trees])
    grouped_leaves = zip(*leaves)
    result_leaves = [jnp.concatenate(l) for l in grouped_leaves]
    return cast(T_co, treedefs[0].unflatten(result_leaves))

def batch_vmap(f: Callable[[KeyArray], T_co], rngs: KeyArray, batch_size: int, progress: bool = False) -> T_co:
    n_batches = int(ceil(rngs.shape[0] / batch_size))
    batch_results: list[T_co] = []
    iterator = _tqdm(range(n_batches)) if progress else range(n_batches)
    for batch_i in iterator:
        batch_rngs = rngs[batch_i * batch_size : (batch_i + 1) * batch_size]
        batch_results.append(vmap(f)(batch_rngs))
    return tree_concatenate(batch_results)

PParams = ParamSpec("PParams")

def to_scalar(f: Callable[PParams, Array]) -> Callable[PParams, Array]:
    @wraps(f)
    def f2(*args: PParams.args, **kwargs: PParams.kwargs) -> Array:
        return f(*args, **kwargs).reshape(())
    return f2

# ====================
# Bootstrap machinery
# ====================
BatchSize = int
ParallelMode = Union[Literal["all"], BatchSize]

@dataclass
class TestResult(Generic[P_]):
    reject_null: bool
    theta_hat: P_
    threshold: float
    test_statistic: float
    bootstrapped_test_stats: Optional[Array]
    bootstrapped_theta_hats: Optional[P_]

class TestStatistic(ABC, Generic[P_]):
    @abstractmethod
    def __call__(self, rng: KeyArray, theta_hat: P_, ys: Array) -> Array:
        ...
    @property
    @abstractmethod
    def name(self) -> str:
        ...

class Bootstrap(Enum):
    PARAMETRIC = "parametric"

def parametric_bootstrap_test(
    rng: KeyArray,
    ys: Array,
    estimator: Estimator[P_],
    null_family: Type[SampleableDist[P_, Array]],
    test_statistic: TestStatistic[P_],
    n_bootstrap_samples: int,
    level: float = 0.05,
    save_null_distribution: bool = False,
    parallel_samples: ParallelMode = "all",
) -> TestResult[P_]:
    rng, rng_input = jax.random.split(rng)
    theta_hat = estimator(rng_input, ys)

    # Sample all bootstrap observations at once then reshape
    rng, rng_input = jax.random.split(rng)
    b_ys = null_family.sample_with_params(
        rng_input, theta_hat, n=ys.shape[0] * n_bootstrap_samples
    ).reshape(n_bootstrap_samples, ys.shape[0], ys.shape[1])

    rng, rng_input = jax.random.split(rng)
    bootstrap_theta_hats, bootstrap_statistics = _make_parametric_bootstrap_samples(
        rng_input,
        b_ys,
        estimator,
        test_statistic,
        n_bootstrap_samples,
        parallel_samples,
    )

    rng, rng_input = jax.random.split(rng)
    statistic_value = test_statistic(rng_input, theta_hat, ys).item()
    crit = jnp.quantile(bootstrap_statistics, 1 - level).item()
    reject_null = statistic_value > crit

    saved_theta_hats = bootstrap_theta_hats if save_null_distribution else None
    saved_stats      = bootstrap_statistics if save_null_distribution else None

    return TestResult(
        reject_null,
        theta_hat,
        crit,
        statistic_value,
        saved_stats,
        saved_theta_hats,
    )

def _make_parametric_bootstrap_samples(
    rng: KeyArray,
    b_ys: Array,
    estimator: Estimator[P_],
    test_statistic: TestStatistic[P_],
    n_bootstrap_samples: int,
    parallel_samples: ParallelMode,
) -> Tuple[P_, Array]:
    batch_size = (n_bootstrap_samples if parallel_samples == "all" else int(parallel_samples))
    if n_bootstrap_samples % batch_size != 0:
        raise ValueError("Bootstrap samples must be divisible by batch size.")
    n_batches = n_bootstrap_samples // batch_size
    n, y_dim = b_ys.shape[1], b_ys.shape[2]
    batched_b_ys = b_ys.reshape(n_batches, batch_size, n, y_dim)

    theta_hats_list = []
    statistics_list = []
    for batch_i in range(n_batches):
        rng_inputs = jax.random.split(rng, num=batch_size)
        theta_hats, statistics = vmap(
            _parametric_bootstrap_sample, in_axes=(0, 0, None, None)
        )(rng_inputs, batched_b_ys[batch_i], estimator, test_statistic)
        theta_hats_list.append(theta_hats)
        statistics_list.append(statistics)
    return tree_concatenate(theta_hats_list), tree_concatenate(statistics_list)

def _parametric_bootstrap_sample(
    rng: KeyArray,
    b_ys: Array,
    estimator: Estimator[P_],
    test_statistic: TestStatistic[P_],
) -> Tuple[P_, Array]:
    rng1, rng2 = jax.random.split(rng, num=2)
    b_ys = cast(Array, b_ys)
    b_theta_hat = estimator(rng1, b_ys)
    return b_theta_hat, test_statistic(rng2, b_theta_hat, b_ys)

# ==========================
# KSD statistic (fast RBF)
# ==========================
class ScoreFunc(Protocol):
    @staticmethod
    def __call__(params: Array, x: Array) -> Array: ...

@partial(jit, static_argnames=("score",))
def h_gram_gaussian_fast(theta: Array, X: Array, l: float, score: ScoreFunc) -> Array:
    # X: (n,d), score(theta, x): (d,)
    n, d = X.shape
    l2 = l * l
    D = X[:, None, :] - X[None, :, :]        # (n,n,d)
    r2 = jnp.sum(D * D, axis=-1)             # (n,n)
    K = jnp.exp(-r2 / (2.0 * l2))            # (n,n)

    S = vmap(lambda x: score(theta, x))(X)   # (n,d)

    # term1: k * (s_i · s_j)
    SS = S @ S.T                              # (n,n)
    term1 = SS * K

    # term2: s_i · ∇_y k =  (s_i · (x_i - x_j)) * k / l^2
    proj_i = jnp.sum(S[:, None, :] * D, axis=-1)  # (n,n)
    term2 = (proj_i / l2) * K

    # term3: s_j · ∇_x k = -(s_j · (x_i - x_j)) * k / l^2
    proj_j = jnp.sum(S[None, :, :] * D, axis=-1)
    term3 = -(proj_j / l2) * K

    # term4: tr(∂^2 k/∂x∂y) = (d/l^2 - r2/l^4) * k
    term4 = ((d / l2) - (r2 / (l2 * l2))) * K

    return term1 + term2 + term3 + term4

@partial(jit, static_argnames=("score",))
def u_stat_gaussian_fast(theta: Array, X: Array, l: float, score: ScoreFunc) -> Array:
    H = h_gram_gaussian_fast(theta, X, l, score)
    n = X.shape[0]
    return (H.sum() - jnp.trace(H)) / (n * (n - 1))

@partial(jit, static_argnames=("score",))
def v_stat_gaussian_fast(theta: Array, X: Array, l: float, score: ScoreFunc) -> Array:
    return h_gram_gaussian_fast(theta, X, l, score).mean()

# Generic (fallback) Stein core using AD
@partial(jit, static_argnames=("kernel", "score"))
def h_gram_generic(kernel: Kernel, score: ScoreFunc, params: Array, ys: Array) -> Array:
    def h(y1: Array, y2: Array) -> Array:
        term1 = kernel(y1, y2) * score(params, y1) @ score(params, y2)
        term2 = score(params, y1) @ grad(kernel, argnums=1)(y1, y2)
        term3 = score(params, y2) @ grad(kernel, argnums=0)(y1, y2)
        term4 = jacfwd(jacfwd(kernel, 0), 1)(y1, y2).trace()
        return term1 + term2 + term3 + term4
    return gram(ys, ys, h)

@partial(jit, static_argnames=("kernel", "score"))
def v_stat(kernel: Kernel, score: ScoreFunc, params: Array, ys: Array) -> Scalar:
    if isinstance(kernel, GaussianKernel):
        return v_stat_gaussian_fast(params, ys, kernel.l, score)
    return h_gram_generic(kernel, score, params, ys).mean()

@partial(jit, static_argnames=("kernel", "score"))
def u_stat(kernel: Kernel, score: ScoreFunc, params: Array, ys: Array) -> Scalar:
    if isinstance(kernel, GaussianKernel):
        return u_stat_gaussian_fast(params, ys, kernel.l, score)
    n = ys.shape[0]
    H = h_gram_generic(kernel, score, params, ys)
    return (H.sum() - jnp.trace(H)) / (n * (n - 1))

# grad wrt theta (still via AD). You can hand-code for MVN if desired.
@partial(jit, static_argnames=("kernel", "score"))
def grad_h_scalar(kernel: Kernel, score: ScoreFunc, theta: Array, x: Array, xp: Array) -> Array:
    def htheta(th):  # scalar
        if isinstance(kernel, GaussianKernel):
            # reuse v_stat with two points for simplicity; fast kernel applies
            return v_stat(kernel, score, th, jnp.stack([x, xp], axis=0)) * 2.0
        else:
            def h(y1, y2):
                term1 = kernel(y1, y2) * score(th, y1) @ score(th, y2)
                term2 = score(th, y1) @ grad(kernel, argnums=1)(y1, y2)
                term3 = score(th, y2) @ grad(kernel, argnums=0)(y1, y2)
                term4 = jacfwd(jacfwd(kernel, 0), 1)(y1, y2).trace()
                return term1 + term2 + term3 + term4
            return h(x, xp)
    return grad(htheta)(theta)

# ===============
# KSD statistic
# ===============
class KSDStatistic(TestStatistic[P_], Generic[P_]):
    def __init__(self, kernel: Kernel, null: Type[UnnormalizedDist]):
        self.kernel = kernel
        self.null = null
    def __call__(self, rng: KeyArray, theta_hat: P_, ys: Array) -> Array:
        return v_stat(self.kernel, self.null.score_with_params, theta_hat, ys)
    @property
    def name(self) -> str: return "ksd"

# ======================
# Gaussian distributions
# ======================
class Gaussian(SampleableDist[Array, Array], NormalizedDist[Array], ExpFamilyDist):
    def __init__(self, loc: Union[Scalar, float], scale: Union[Scalar, float]) -> None:
        self.loc = jnp.array(loc, dtype=jnp.float32)
        self.scale = jnp.array(scale, dtype=jnp.float32)

    def prob(self, x: Array) -> Array:
        return norm.pdf(x, loc=self.loc, scale=self.scale)

    @staticmethod
    def sample_with_params(rng: KeyArray, params: Array, n: int) -> Array:
        return _sample(rng, loc=params[0], scale=params[1], n=n)

    @staticmethod
    def score_with_params(params: Array, x: Scalar) -> Array:
        return _score(x, loc=params[0], scale=params[1])

    @staticmethod
    def log_prob_with_params(params: Array, x: Scalar) -> Array:
        return norm.logpdf(x, params[0], params[1])

    def get_params(self) -> Array:
        return jnp.array([self.loc, self.scale], dtype=jnp.float32)

    @staticmethod
    def natural_parameter(params: Array) -> Array:
        loc, scale = params[0], params[1]
        return jnp.array([loc / scale, -1.0 / (2.0 * scale)], dtype=jnp.float32)

    @staticmethod
    def natural_parameter_inverse(eta_val: Array) -> Array:
        loc = -0.5 * eta_val[0] / eta_val[1]
        scale = jnp.sqrt(1.0 / (-2.0 * eta_val[1]))
        return jnp.array([loc, scale], dtype=jnp.float32)

    @staticmethod
    def sufficient_statistic(x: Array) -> Array:
        return jnp.concatenate([x, x**2], axis=0)

    @staticmethod
    def b(x: Array) -> Array:
        return jnp.zeros(shape=(), dtype=x.dtype)

def gaussian_fixed_scale(scale: Union[float, Scalar]):
    s = jnp.array(scale, dtype=jnp.float32)
    class GaussianFixedScale(SampleableDist[Array, Array], NormalizedDist[Array], ExpFamilyDist):
        def __init__(self, loc: Scalar) -> None:
            self.loc = jnp.array([loc], dtype=jnp.float32)
        def prob(self, x: Array) -> Array:
            return norm.pdf(x, loc=self.loc, scale=s)
        @staticmethod
        def sample_with_params(rng: KeyArray, params: Array, n: int) -> Array:
            return _sample(rng, loc=params[0], scale=s, n=n)
        @staticmethod
        def score_with_params(params: Array, x: Scalar) -> Array:
            return _score(x, loc=params[0], scale=s)
        @staticmethod
        def log_prob_with_params(params: Array, x: Scalar) -> Array:
            return norm.logpdf(x, params[0], s)
        def get_params(self) -> Array: return self.loc
        @staticmethod
        def natural_parameter(params: Array) -> Array:
            mean = params[0]; return mean / (s**2)
        @staticmethod
        def natural_parameter_inverse(eta_val: Array) -> Array:
            mean = -0.5 * eta_val[0] / eta_val[1]
            std = jnp.sqrt(1 / (-2 * eta_val[1]))
            return jnp.array([mean, std], dtype=jnp.float32)
        @staticmethod
        def sufficient_statistic(x: Array) -> Array:
            return jnp.concatenate([x, x**2], axis=0)
        @staticmethod
        def b(x: Array) -> Array:
            return jnp.zeros(shape=(), dtype=x.dtype)
    return GaussianFixedScale

@partial(jit, static_argnames=("n",))
def _sample(rng: KeyArray, loc: Scalar, scale: Scalar, n: int) -> Array:
    return (loc + scale * jax.random.normal(rng, shape=(n, 1))).astype(jnp.float32)

def _score(x: Array, loc: Scalar, scale: Scalar) -> Array:
    return grad(to_scalar(norm.logpdf), argnums=0)(x.reshape(x.shape[0]), loc, scale)

# ======================================
# KSD estimator for exponential families
# ======================================
@partial(jit, static_argnames=("kernel", "dist_family"))
def ksd_estimator_exp_family(
    kernel: Kernel,
    dist_family: Type[ExpFamilyDist],
    ys: Array,
) -> Array:
    eta_inv = dist_family.natural_parameter_inverse
    t = dist_family.sufficient_statistic
    b = dist_family.b

    def Lambda(y1: Array, y2: Array) -> Array:
        return kernel(y1, y2) * atleast_2d(jacfwd(t)(y1)) @ atleast_2d(jacfwd(t)(y2)).T

    def nu(y1: Array, y2: Array) -> Array:
        term1 = kernel(y1, y2) * jacfwd(b)(y1) @ jacfwd(t)(y2).T
        term2 = atleast_2d(jacfwd(t)(y1)) @ jacfwd(kernel, argnums=1)(y1, y2)
        term3 = kernel(y1, y2) * jacfwd(b)(y2) @ jacfwd(t)(y1).T
        term4 = atleast_2d(jacfwd(t)(y2)) @ jacfwd(kernel, argnums=0)(y1, y2)
        return term1 + term2 + term3 + term4

    big_lambda_n = gram(ys, ys, Lambda).mean(0).mean(0)
    big_lambda_n = big_lambda_n + jnp.eye(big_lambda_n.shape[0], dtype=big_lambda_n.dtype) * 1e-4
    nu_n = gram(ys, ys, nu).mean(0).mean(0)

    eta_estimate = jnp.linalg.solve(big_lambda_n, -0.5 * nu_n)
    theta_estimate = eta_inv(eta_estimate)
    return theta_estimate

# ======================================
# Composite ~KSD bootstrap (U-statistic)
# ======================================
@partial(jit, static_argnames=("f",))
def U_n_from_f(X: Array, f: Callable[[Array, Array], Array]) -> Array:
    n = X.shape[0]
    F = vmap(lambda xi: vmap(lambda xj: f(xi, xj))(X))(X)  # (n,n) or (n,n,p)
    if F.ndim == 2:
        mask = 1.0 - jnp.eye(n, dtype=F.dtype)
        return (F * mask).sum() / (n * (n - 1))
    else:
        mask = 1.0 - jnp.eye(n, dtype=F.dtype)
        return (F * mask[..., None]).sum(axis=(0, 1)) / (n * (n - 1))

@partial(jit, static_argnames=("kernel", "score"))
def centered_core_matrix_for_theta(kernel: Kernel, score: ScoreFunc, theta: Array, X: Array) -> Array:
    H = (h_gram_gaussian_fast(theta, X, kernel.l, score)
         if isinstance(kernel, GaussianKernel)
         else h_gram_generic(kernel, score, theta, X))
    row_mean = H.mean(axis=1, keepdims=True)
    col_mean = H.mean(axis=0, keepdims=True)
    v_statistic = H.mean(axis=(0, 1), keepdims=True)
    return H - row_mean - col_mean + v_statistic

def KSD_test(
    kernel: Kernel,
    score: ScoreFunc,
    rng: KeyArray,
    X: Array,
    B: int,
    level: float,
    estimator_fn: Callable[[KeyArray, Array], Array],  # (rng, sample) -> theta
):
    """
    Composite ~KSD (U-stat) with empirical centering + gradient correction.
    This optimized version precomputes bootstrap indices on the host.
    """
    n = X.shape[0]
    rng, r_est = jax.random.split(rng)
    theta_hat = estimator_fn(r_est, X)

    # Precompute bootstrap indices + estimator keys
    rng, r_idx, r_keys = jax.random.split(rng, 3)
    idx_mat = jax.random.randint(r_idx, (B, n), minval=0, maxval=n)
    keys_est = jax.random.split(r_keys, B)
    return _KSD_test_preindexed(kernel, score, X, level, estimator_fn, theta_hat, idx_mat, keys_est)

@partial(jit, static_argnames=("kernel", "score", "estimator_fn"))
def _KSD_test_preindexed(
    kernel: Kernel,
    score: ScoreFunc,
    X: Array,
    level: float,
    estimator_fn: Callable[[KeyArray, Array], Array],
    theta_hat: Array,
    idx_mat: Array,     # (B, n)
    keys_est: Array,    # (B,)
):
    n = X.shape[0]
    T_obs = (u_stat_gaussian_fast(theta_hat, X, kernel.l, score)
             if isinstance(kernel, GaussianKernel)
             else u_stat(kernel, score, theta_hat, X))

    grad_h_hat = tree_util.Partial(lambda x, y: grad_h_scalar(kernel, score, theta_hat, x, y))
    U_grad_orig = _u_offdiag_vec(X, grad_h_hat)  # (p,)

    def one_bootstrap(idx: Array, k_est: KeyArray) -> Array:
        Xb = X[idx]
        theta_star = estimator_fn(k_est, Xb)
        Hc_star = centered_core_matrix_for_theta(kernel, score, theta_star, X)
        Hc_sub  = Hc_star[idx][:, idx]
        U_star_c = (Hc_sub.sum() - jnp.trace(Hc_sub)) / (n * (n - 1))
        grad_h_star = tree_util.Partial(lambda x, y: grad_h_scalar(kernel, score, theta_star, x, y))
        U_grad_star = _u_offdiag_vec(Xb, grad_h_star)
        corr = jnp.dot(theta_star - theta_hat, U_grad_star - U_grad_orig)
        return U_star_c + corr

    T_b_vals = lax.map(lambda pair: one_bootstrap(pair[0], pair[1]), (idx_mat, keys_est))
    crit = jnp.quantile(T_b_vals, 1.0 - level)
    return {
        "crit": crit,
        "T_obs": T_obs,
        "reject": T_obs > crit,
        "theta_hat": theta_hat,
        "KSD-tildes": T_b_vals,
    }

@partial(jit, static_argnames=("f_xy",))
def _u_offdiag_vec(X: Array, f_xy: Callable[[Array, Array], Array]) -> Array:
    F = vmap(lambda x: vmap(lambda y: f_xy(x, y))(X))(X)  # (n,n,p)
    n = X.shape[0]
    mask = 1.0 - jnp.eye(n, dtype=F.dtype)
    return (F * mask[..., None]).sum(axis=(0, 1)) / (n * (n - 1))

@partial(jit, static_argnames=("kernel", "null_family"))
def _estimator_adapter(kernel: Kernel, null_family: Type[UnnormalizedDist], rng: KeyArray, sample: Array) -> Array:
    del rng
    return ksd_estimator_exp_family(kernel, null_family, sample)

def run_ksd_test(
    rng: KeyArray,
    X: Array,
    kernel: Kernel,
    null_family: Type[UnnormalizedDist],
    *,
    B: int = 400,
    level: float = 0.05,
):
    estimator_fn = lambda r, samp: _estimator_adapter(kernel, null_family, r, samp)
    return KSD_test(kernel, null_family.score_with_params, rng, X, B, level, estimator_fn)

# ===============================
# MVN exponential-family (Cholesky)
# ===============================
from jax.nn import softplus
from jax.scipy.linalg import solve_triangular

def _vech_indices(d: int):
    return jnp.tril_indices(d)

def vech_sym(M: Array) -> Array:
    i, j = _vech_indices(M.shape[0])
    return M[i, j]

def sym_from_vech(v: Array, d: int) -> Array:
    i, j = _vech_indices(d)
    M = jnp.zeros((d, d), dtype=v.dtype)
    M = M.at[i, j].set(v)
    return 0.5 * (M + M.T - jnp.diag(jnp.diag(M)))

def _tril_indices_full(d: int):
    return jnp.tril_indices(d)

def _tril_size(d: int) -> int:
    return d * (d + 1) // 2

def _pack_tril(L: Array) -> Array:
    i, j = _tril_indices_full(L.shape[0])
    return L[i, j]

def _unpack_tril(v: Array, d: int) -> Array:
    i, j = _tril_indices_full(d)
    L = jnp.zeros((d, d), dtype=v.dtype)
    return L.at[i, j].set(v)

def _inv_softplus(x: Array, eps: float = 1e-12) -> Array:
    return jnp.log(jnp.expm1(jnp.maximum(x, eps)))

def mvn_full_family_chol(d: int):
    ii = jnp.diag_indices(d)
    def _unpack_params(theta: Array) -> Tuple[Array, Array]:
        mu = theta[:d]
        L_uncon = theta[d:]
        L = _unpack_tril(L_uncon, d)
        L = L.at[ii].set(softplus(L[ii]) + 1e-6)
        return mu, L

    class MVNFullChol(ExpFamilyDist, UnnormalizedDist[Array]):
        @staticmethod
        def score_with_params(params: Array, x: Array) -> Array:
            mu, L = _unpack_params(params)
            b = (mu - x)
            y = solve_triangular(L, b, lower=True)
            s = solve_triangular(L.T, y, lower=False)
            return s

        @staticmethod
        def unnorm_log_prob_with_params(params: Array, x: Array) -> Array:
            mu, L = _unpack_params(params)
            Sigma = L @ L.T + 1e-6 * jnp.eye(d, dtype=L.dtype)
            return multivariate_normal.logpdf(x, mean=mu, cov=Sigma)

        @staticmethod
        def log_prob_with_params(params: Array, x: Array) -> Array:
            mu, L = _unpack_params(params)
            Sigma = L @ L.T + 1e-6 * jnp.eye(d, dtype=L.dtype)
            return multivariate_normal.logpdf(x, mean=mu, cov=Sigma)

        @staticmethod
        def sufficient_statistic(x: Array) -> Array:
            xxT = jnp.outer(x, x)
            return jnp.concatenate([x, vech_sym(xxT)], axis=0)

        @staticmethod
        def b(x: Array) -> Array:
            return jnp.zeros((), dtype=x.dtype)

        @staticmethod
        def natural_parameter(params: Array) -> Array:
            mu, L = _unpack_params(params)
            Sigma = L @ L.T + 1e-6 * jnp.eye(d, dtype=L.dtype)
            P = jnp.linalg.inv(Sigma)
            eta1 = P @ mu
            eta2 = -0.5 * P
            return jnp.concatenate([eta1, vech_sym(eta2)], axis=0)

        @staticmethod
        def natural_parameter_inverse(eta_val: Array) -> Array:
            eta1 = eta_val[:d]
            vech_eta2 = eta_val[d:]
            Eta2 = sym_from_vech(vech_eta2, d)
            Eta2 = 0.5 * (Eta2 + Eta2.T)
            P = -2.0 * Eta2 + 1e-6 * jnp.eye(d, dtype=Eta2.dtype)
            Sigma = jnp.linalg.inv(P)
            L = jnp.linalg.cholesky(Sigma + 1e-6 * jnp.eye(d, dtype=Sigma.dtype))
            mu = jnp.linalg.solve(P, eta1)
            L_uncon = L.at[ii].set(_inv_softplus(L[ii]))
            theta = jnp.concatenate([mu, _pack_tril(L_uncon)], axis=0)
            return theta

        @staticmethod
        def sample_with_params(rng: KeyArray, params: Array, n: int) -> Array:
            mu, L = _unpack_params(params)
            z = jax.random.normal(rng, shape=(n, d))
            return (mu + z @ L.T).astype(jnp.float32)

        def get_params(self) -> Array:
            raise NotImplementedError("Use static methods with explicit params.")
    return MVNFullChol

# RBF kernel lengthscale rule: l = sqrt(d)  (tweak *0.5 if you want smaller lengthscales)
def make_kernel_for_dim(d: int) -> GaussianKernel:
    return GaussianKernel(l=float(np.sqrt(d) * 0.2))   #TODO use same as in null hypothesis  0.2

# ======================
# BFM (CuPy implementation)
# ======================
def kern_k(x, y=None, bw=1.0, amp=1.0):
    x = cp.asarray(x, dtype=cp.float32)
    if x.ndim == 1: x = x.reshape(-1, 1)
    y = x if y is None else cp.asarray(y, dtype=cp.float32)
    if y.ndim == 1: y = y.reshape(-1, 1)
    x_norm = cp.sum(x ** 2, axis=1).reshape(-1, 1)
    y_norm = cp.sum(y ** 2, axis=1).reshape(1, -1)
    d2 = x_norm + y_norm - 2 * x.dot(y.T)
    return amp * cp.exp(-(1.0 / (2.0 * bw * bw)) * d2)

def MMD_stand(X, Y, bw, unbiased=True):
    Kxx = kern_k(X, X, bw=bw)
    Kyy = kern_k(Y, Y, bw=bw)
    Kxy = kern_k(X, Y, bw=bw)
    n = X.shape[0]
    if unbiased:
        cp.fill_diagonal(Kxx, 0.0)
        cp.fill_diagonal(Kyy, 0.0)
        cp.fill_diagonal(Kxy, 0.0)
        term3 = 2 * cp.sum(Kxy) / (n * (n - 1))
        term1 = cp.sum(Kxx) / (n * (n - 1))
        term2 = cp.sum(Kyy) / (n * (n - 1))
    else:
        term3 = 2 * cp.sum(Kxy) / (n * n)
        term1 = cp.sum(Kxx) / (n * n)
        term2 = cp.sum(Kyy) / (n * n)
    return term1 + term2 - term3

def bfm_u_stat(X, Y, bw):
    n = X.shape[0]
    m = n // 2
    X1, X2 = X[::2], X[1::2]
    Y1, Y2 = Y[::2], Y[1::2]
    def rbf(A, B):
        A_norm = cp.sum(A ** 2, axis=1).reshape(-1, 1)
        B_norm = cp.sum(B ** 2, axis=1).reshape(1, -1)
        d2 = A_norm + B_norm - 2 * A.dot(B.T)
        return cp.exp(-d2 / (2.0 * bw * bw))
    K_x1x1 = rbf(X1, X1)
    K_y1y1 = rbf(Y1, Y1)
    K_y2x2 = rbf(Y2, X2)
    H = K_x1x1 + K_y1y1 - K_y2x2 - K_y2x2.T
    upper_sum = cp.sum(cp.triu(H, k=1))
    return 2.0 * upper_sum / (m * (m - 1))

def stand_h_vec(X, Y, bw):
    return kern_k(X, X, bw=bw) + kern_k(Y, Y, bw=bw) - kern_k(X, Y, bw=bw) - kern_k(X, Y, bw=bw).T

def sigma_spec_a(d, n, X, Y, bw, mmd_hat):
    H = stand_h_vec(X, Y, bw=bw)
    h1 = cp.mean(H, axis=1) - mmd_hat
    s_a_squared = 4.0 * cp.var(h1, ddof=1)
    return {"s.a": cp.sqrt(s_a_squared), "h1.a": h1}

def new_h_vec(Z, bw):
    d = Z.shape[1] // 4
    x1 = Z[:, :d]; y1 = Z[:, d:2*d]; x2 = Z[:, 2*d:3*d]; y2 = Z[:, 3*d:]
    return kern_k(x1, x1, bw=bw) + kern_k(y1, y1, bw=bw) - kern_k(x2, y2, bw=bw) - kern_k(x2, y2, bw=bw).T

def sigma_spec_q(d, n, X, Y, bw, mmd_hat):
    idx = cp.arange(0, n, 2)
    Z = cp.hstack([X[idx], Y[idx], X[idx+1], Y[idx+1]])
    H = new_h_vec(Z, bw=bw)
    h1 = cp.mean(H, axis=1) - mmd_hat
    s_q_squared = 8.0 * cp.var(h1, ddof=1)
    return {"s.q": cp.sqrt(s_q_squared), "h1.q": h1}

def BFM_test(d, n, epsilon, X_cp, Y_cp, Y2_cp=None, bw=1.0, model_selection=False, level=0.05):
    def _run_test(eps):
        X = cp.asarray(X_cp, dtype=cp.float32)
        Y = cp.asarray(Y_cp, dtype=cp.float32)
        if not model_selection:
            MMD_hat = MMD_stand(X, Y, bw, unbiased=True)
            MMD_q = bfm_u_stat(X, Y, bw)
            MMD_eps = MMD_hat + eps * MMD_q
            out_a = sigma_spec_a(d, n, X, Y, bw, MMD_hat)
            out_q = sigma_spec_q(d, n, X, Y, bw, MMD_hat)
            s_a = out_a["s.a"]; s_q = out_q["s.q"]
            sd_mmd_eps = s_a + eps * s_q
            z = float(cp.sqrt(n) * MMD_eps / sd_mmd_eps)
            zq = float(cp.sqrt(n) * MMD_q / s_q)
            th = float(scipy_norm.ppf(1 - level / 2))
            return {
                "spec.test.stat.1": z,
                "spec.reject.1": z > th,
                "MMD.eps.1": float(MMD_eps),
                "MMD.hat.1": float(MMD_hat),
                "MMD.q.1": float(MMD_q),
                "spec.test.stat.q.1": zq,
                "spec.reject.q.1": zq > th,
                "s.a.1": float(s_a),
                "s.q.1": float(s_q),
            }
        raise NotImplementedError("model_selection=True path not implemented.")
    if np.isscalar(epsilon):
        return _run_test(float(epsilon))
    else:
        return [_run_test(float(eps)) for eps in epsilon]

# ===============
# GMM alternative
# ===============
def _sample_gmm(key: KeyArray, n: int, d: int, mu: float, sigma: float = 1.0) -> Array:
    key_u, key_z = jax.random.split(key)
    signs = 2 * jax.random.bernoulli(key_u, 0.5, shape=(n,)) - 1
    means = jnp.zeros((n, d), dtype=jnp.float32)
    means = means.at[:, 0].set(mu * signs)
    z = jax.random.normal(key_z, shape=(n, d))
    return (means + sigma * z).astype(jnp.float32)

# -------------
# BFM wrapper
# -------------
def _fit_mle_gaussian_np(X_np: np.ndarray):
    X_np = np.asarray(X_np)
    if X_np.ndim == 1: X_np = X_np.reshape(-1, 1)
    mu_hat = X_np.mean(axis=0)
    Sigma_hat = np.atleast_2d(np.cov(X_np, rowvar=False, bias=False))
    eps = 1e-8
    d = Sigma_hat.shape[0]
    Sigma_hat = Sigma_hat + eps * np.eye(d, dtype=Sigma_hat.dtype)
    return mu_hat.astype(np.float32), Sigma_hat.astype(np.float32)

def _bw_from_null_fit(Sigma_hat_np: np.ndarray) -> float:
    return float(np.sqrt(np.trace(Sigma_hat_np)))

def bfm_reject_one(key: KeyArray, X: Array, *, level: float = 0.05) -> tuple[bool, dict]:
    X = X.astype(jnp.float32)
    X_cp = jax_to_cupy(X)  # robust transfer
    n, d = X.shape

    # fit null on host (small)
    mu_hat, Sigma_hat = _fit_mle_gaussian_np(np.array(X))
    bw = _bw_from_null_fit(Sigma_hat)

    # simulate Y from fitted null using JAX then transfer once
    key_y = jax.random.split(key, 1)[0]
    L = np.linalg.cholesky(Sigma_hat).astype(np.float32)
    z = jax.random.normal(key_y, shape=(n, d)).astype(jnp.float32)
    Y = (jnp.array(mu_hat) + jnp.array(z @ jnp.array(L.T))).astype(jnp.float32)
    Y_cp = jax_to_cupy(Y)

    eps_n = float(n ** (-1.0 / 2.5))
    out = BFM_test(d=d, n=n, epsilon=eps_n, X_cp=X_cp, Y_cp=Y_cp,
                   bw=bw, model_selection=False, level=level)
    return bool(out["spec.reject.1"]), out

# ===========================
# MMD statistic (parametric)
# ===========================
@dataclass(frozen=True)
class MMDStatistic(TestStatistic, Generic[P_]):
    kernel: Kernel
    null_dist: type[SampleableDist[P_, Array]]

    def __call__(self, rng: KeyArray, theta_hat: P_, ys: Array) -> Array:
        xs = self.null_dist.sample_with_params(rng, theta_hat, ys.shape[0])
        return mmd_v_stat(xs, ys, self.kernel)

    @property
    def name(self) -> str: return "mmd"

@partial(jit, static_argnames=("kernel",))
def mmd_v_stat(xs: Array, ys: Array, kernel: Kernel) -> Array:
    return mmd_h_gram(xs, ys, kernel).mean()

@partial(jit, static_argnames=("kernel",))
def mmd_h_gram(xs: Array, ys: Array, kernel: Kernel) -> Array:
    assert xs.shape[0] == ys.shape[0]
    K_xx = gram(xs, xs, kernel)
    K_yy = gram(ys, ys, kernel)
    K_xy = gram(xs, ys, kernel)
    return K_yy + K_xx - K_xy - K_xy.T

# MVN MLE estimator (parametric bootstrap for MMD)
class MVNMLEEstimator(Estimator[Array]):
    def __init__(self, d: int) -> None:
        self.d = d
        self.ii = jnp.diag_indices(d)

    def __call__(self, rng: KeyArray, ys: Array) -> Array:
        del rng
        ys = ys.astype(jnp.float32)
        n = ys.shape[0]
        mu = ys.mean(axis=0)
        Xc = ys - mu
        S = (Xc.T @ Xc) / jnp.maximum(n - 1, 1)
        S = S + 1e-6 * jnp.eye(self.d, dtype=S.dtype)
        L = jnp.linalg.cholesky(S)
        L_uncon = L.at[self.ii].set(_inv_softplus(L[self.ii]))
        theta = jnp.concatenate([mu, _pack_tril(L_uncon)], axis=0)
        return theta.astype(jnp.float32)

    @property
    def name(self) -> str: return "mle"

# Wrappers
def ksd_reject_one(key: KeyArray, X: Array, *, d: int, B: int = 200, level: float = 0.05) -> tuple[bool, dict]:
    kernel = make_kernel_for_dim(d)
    MVN = mvn_full_family_chol(d)
    est_fn = lambda r, Y: ksd_estimator_exp_family(kernel, MVN, Y)
    res = KSD_test(kernel, MVN.score_with_params, key, X, B, level, est_fn)
    return bool(res["reject"]), res

def ksd_parametric_reject_one(key: KeyArray, X: Array, *, d: int, B: int = 200, level: float = 0.05, parallel: ParallelMode = 128):
    kernel = make_kernel_for_dim(d)
    MVN = mvn_full_family_chol(d)
    estimator = KSDAnalyticEstimator(kernel, MVN)
    stat      = KSDStatistic(kernel, MVN)
    res = parametric_bootstrap_test(
        key, X, estimator=estimator, null_family=MVN, test_statistic=stat,
        n_bootstrap_samples=B, level=level, parallel_samples=parallel
    )
    return bool(res.reject_null), res

def mmd_parametric_reject_one(key: KeyArray, X: Array, *, d: int, B: int = 200, level: float = 0.05, parallel: ParallelMode = 128):
    kernel = make_kernel_for_dim(d)
    MVN = mvn_full_family_chol(d)
    estimator = MVNMLEEstimator(d)
    stat      = MMDStatistic(kernel=kernel, null_dist=MVN)
    res = parametric_bootstrap_test(
        key, X, estimator=estimator, null_family=MVN,
        test_statistic=stat, n_bootstrap_samples=B,
        level=level, parallel_samples=parallel
    )
    return bool(res.reject_null), res

class KSDAnalyticEstimator(Estimator[Array]):
    def __init__(self, kernel: Kernel, dist_family: ExpFamilyDist) -> None:
        self.kernel = kernel
        self.dist_family = dist_family
    def __call__(self, rng: KeyArray, ys: Array) -> Array:
        del rng
        return ksd_estimator_exp_family(self.kernel, self.dist_family, ys).astype(jnp.float32)
    @property
    def name(self) -> str: return "ksd"

# =========================
# Power vs μ: GMM alternative
# =========================
def power_gmm_vs_bfm(
    *,
    d_list=(1, 2, 5, 10, 20),
    n_list=(100, 300, 500),
    mu_grid=np.linspace(0.0, 3.0, 7),
    runs: int = 200,
    level: float = 0.05,
    B_ksd: int = 200,             # ~KSD composite bootstrap
    B_param: int | None = None,   # parametric bootstrap draws (KSD/MMD)
    sigma_by_d: float | dict[int, float] = 1.0,
    seed: int = 123,
    progress: bool = True,
    parallel_param_boot: ParallelMode = 128,   # batch size for param boot
) -> pd.DataFrame:
    """
    For each (d, n, μ): simulate X ~ GMM_μ and run on the SAME X:
      - KSD (~U composite bootstrap)
      - KSD (parametric bootstrap)
      - MMD (parametric bootstrap)
      - BFM (specification test)
    No wild bootstrap is used.
    """
    if B_param is None:
        B_param = B_ksd

    key = jax.random.PRNGKey(seed)
    rows = []

    for d in d_list:
        sigma = float(sigma_by_d[d]) if isinstance(sigma_by_d, dict) else float(sigma_by_d)
        kernel = make_kernel_for_dim(d)
        MVN    = mvn_full_family_chol(d)
        est_ksd_analytic = KSDAnalyticEstimator(kernel, MVN)
        stat_ksd         = KSDStatistic(kernel, MVN)
        est_mle          = MVNMLEEstimator(d)
        mmd_stat         = MMDStatistic(kernel=kernel, null_dist=MVN)

        for n in n_list:
            powers_ksd_u, powers_bfm = [], []
            powers_ksd_param, powers_mmd_param = [], []

            iterator = _tqdm(mu_grid, desc=f"d={d} n={n}", disable=not progress)
            for mu in iterator:
                rej_ksd_u = rej_bfm = 0
                rej_ksd_pb = rej_mmd_pb = 0

                for _ in range(runs):
                    key, k_samp, k_ksd_u, k_bfm, k_ksd_pb, k_mmd_pb = jax.random.split(key, 6)
                    X = _sample_gmm(k_samp, n=n, d=d, mu=float(mu), sigma=sigma)

                    # 1) KSD (~U)
                    r_ksd_u, _ = ksd_reject_one(k_ksd_u, X, d=d, B=B_ksd, level=level)
                    rej_ksd_u += int(r_ksd_u)

                    # 2) BFM
                    r_bfm, _ = bfm_reject_one(k_bfm, X, level=level)
                    rej_bfm += int(r_bfm)

                    # 3) KSD parametric
                    res_ksd_pb = parametric_bootstrap_test(
                        k_ksd_pb, X, estimator=est_ksd_analytic, null_family=MVN,
                        test_statistic=stat_ksd, n_bootstrap_samples=B_param,
                        level=level, parallel_samples=parallel_param_boot
                    )
                    rej_ksd_pb += int(res_ksd_pb.reject_null)

                    # 4) MMD parametric
                    res_mmd_pb = parametric_bootstrap_test(
                        k_mmd_pb, X, estimator=est_mle, null_family=MVN,
                        test_statistic=mmd_stat, n_bootstrap_samples=B_param,
                        level=level, parallel_samples=parallel_param_boot
                    )
                    rej_mmd_pb += int(res_mmd_pb.reject_null)

                def _p_se(r):
                    p = r / runs
                    return p, float(np.sqrt(p * (1 - p) / runs))

                p_ksd_u, se_ksd_u       = _p_se(rej_ksd_u)
                p_bfm,   se_bfm         = _p_se(rej_bfm)
                p_ksd_pb, se_ksd_pb     = _p_se(rej_ksd_pb)
                p_mmd_pb, se_mmd_pb     = _p_se(rej_mmd_pb)

                powers_ksd_u.append(p_ksd_u)
                powers_bfm.append(p_bfm)
                powers_ksd_param.append(p_ksd_pb)
                powers_mmd_param.append(p_mmd_pb)

                rows += [
                    dict(test="KSD (~U bootstrap)",          d=d, n=n, mu=float(mu), power=float(p_ksd_u),  se=se_ksd_u,  B=B_ksd,   alpha=level),
                    dict(test="BFM",                          d=d, n=n, mu=float(mu), power=float(p_bfm),    se=se_bfm,    bw_rule="sqrt(trace(Sigma_hat))", alpha=level),
                    dict(test="KSD (parametric bootstrap)",   d=d, n=n, mu=float(mu), power=float(p_ksd_pb), se=se_ksd_pb, B=B_param, alpha=level),
                    dict(test="MMD (parametric bootstrap)",   d=d, n=n, mu=float(mu), power=float(p_mmd_pb), se=se_mmd_pb, B=B_param, alpha=level),
                ]

            print(f"\n== d={d}, n={n} ==")
            print("KSD ~U power:", np.array(powers_ksd_u))
            print("BFM power:",    np.array(powers_bfm))
            print("KSD param power:", np.array(powers_ksd_param))
            print("MMD param power:", np.array(powers_mmd_param))

    df = pd.DataFrame(rows)
    return df

# =========
# __main__
# =========
if __name__ == "__main__":
    df_power = power_gmm_vs_bfm(
        d_list=(1, 2, 4),
        n_list=(100, 200, 300, 400, 500),
        mu_grid=np.linspace(1, 4, 3),
        runs=1,
        level=0.05,
        B_ksd=200,
        B_param=200,
        sigma_by_d={1: 1.0, 2: 1.0, 4: 1.0},
        seed=11,
        progress=True,
        parallel_param_boot=10,
    )
    print("\nSummary head:")
    print(df_power.head())
