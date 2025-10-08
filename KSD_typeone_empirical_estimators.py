from functools import partial,wraps
from typing import Generic, Protocol, Type, TypeVar,Callable, Iterable, Optional,Literal,Tuple,Union,cast,ParamSpec
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from math import ceil

import jax
import jax.numpy as jnp
from jax import Array, grad, jacfwd, jacrev, jit, vmap,tree_util,random
from jax.numpy import atleast_2d
from jax.scipy.stats import multivariate_normal, norm
from jax.random import PRNGKey
from tqdm import tqdm
import matplotlib.pyplot as plt
from jax import lax  #for memory

import numpy as np  #TODO delete when not needed anymore
import math

try:                                   # JAX < 0.4.24  authors had old version
    from jax.random import KeyArray
except (ImportError, AttributeError):  # JAX ≥ 0.4.24
    # Runtime alias so the rest of the code keeps working.
    # Optional: use typing.NewType for stricter static checks.
    KeyArray = jax.Array               # type: ignore[assignment]




### distributions
X = TypeVar("X")
P = TypeVar("P")


class Distribution(ABC, Generic[P]):
    @abstractmethod
    def get_params(self) -> P:
        pass


class SampleableDist(Distribution[P], ABC, Generic[P, X]):
    def sample(self, rng: KeyArray, n: int) -> X:
        return self.sample_with_params(rng, self.get_params(), n)

    @staticmethod
    @abstractmethod
    def sample_with_params(rng: KeyArray, params: P, n: int) -> X:
        pass


class UnnormalizedDist(Distribution[P], ABC, Generic[P]):
    def score(self, x: Array) -> Array:
        return self.score_with_params(self.get_params(), x)

    @staticmethod
    @abstractmethod
    def score_with_params(params: P, x: Array) -> Array:
        pass

    def unnorm_log_prob(self, x: Array) -> Array:
        return self.unnorm_log_prob_with_params(self.get_params(), x)

    @classmethod
    @abstractmethod
    def unnorm_log_prob_with_params(cls, params: P, x: Array) -> Array:
        pass


class NormalizedDist(UnnormalizedDist[P], ABC, Generic[P]):
    def log_prob(self, x: Array) -> Array:
        return self.log_prob_with_params(self.get_params(), x)

    @staticmethod
    @abstractmethod
    def log_prob_with_params(params: P, x: Array) -> Array:
        pass

    @classmethod
    def unnorm_log_prob_with_params(cls, params: P, x: Array) -> Array:
        return cls.log_prob_with_params(params, x)


class ExpFamilyDist(Protocol):
    @staticmethod
    @abstractmethod
    def natural_parameter(params: Array) -> Array:
        """param space -> R^k"""
        pass

    @staticmethod
    @abstractmethod
    def natural_parameter_inverse(eta_val: Array) -> Array:
        """R^k -> param space"""
        pass

    @staticmethod
    @abstractmethod
    def sufficient_statistic(x: Array) -> Array:
        """R^d -> R^k"""
        pass

    @staticmethod
    @abstractmethod
    def b(x: Array) -> Array:
        """R^d -> R"""
        pass


class SampleableAndNormalizedDist(
    SampleableDist[P, X], NormalizedDist[P], ABC, Generic[P, X]
):
    pass


class SampleableAndUnnormalizedDist(
    SampleableDist[P, X], UnnormalizedDist[P], ABC, Generic[P, X]
):
    pass

### estimators
P = TypeVar("P")

class Estimator(ABC, Generic[P]):
    @abstractmethod
    def __call__(self, rng: KeyArray, ys: Array) -> P:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


@dataclass
class TrueEstimator(Estimator[P]):
    """An estimator that just returns a given value, useful for debugging.

    You can use this to turn a composite test into a non-composite test, by creating an
    estimator that just returns the true value of the parameter.
    """

    theta: P

    def __call__(self, rng: KeyArray, ys: Array) -> P:
        return self.theta

    @property
    def name(self) -> str:
        return "true"


Scalar=Array

### kernels
KernelLike = Callable[[Array, Array], Array]

class Kernel(ABC):
    @abstractmethod
    def __call__(self, x1: Array, x2: Array) -> Scalar:
        pass


# It's important that frozen,eq=True so dataclasses implements a hash function, which
# avoids JAX recompiling unecessarily if the kernel is the same.
# Use Chex dataclasses to avoid recompiling when only a kernel parameter (e.g. the
# lengthscale) changes.
@dataclass(frozen=True, eq=True)
class GaussianKernel(Kernel):
    l: float

    def __call__(self, x1: Array, x2: Array) -> Scalar:
        return jnp.exp(-((x1 - x2) ** 2).sum() / (2* self.l**2))   #TODO previously jnp.exp(-((x1 - x2) ** 2).sum() / (2 * self.l**2))


@dataclass(frozen=True, eq=True)
class IMQKernel(Kernel):
    l: float
    gamma: float = 0.5

    def __post_init__(self) -> None:
        assert self.l >= 0.0
        assert 0.0 <= self.gamma and self.gamma <= 1.0

    def __call__(self, x1: Array, x2: Array) -> Scalar:
        return (1 + ((x1 - x2) ** 2).sum() / (2 * self.l**2)) ** -self.gamma


class SumKernel(Kernel):
    def __init__(self, kernels: Iterable[Kernel]) -> None:
        self.kernels = tuple(kernels)

    def __call__(self, x1: Array, x2: Array) -> Array:
        return jnp.array([k(x1, x2) for k in self.kernels]).sum()

    def __hash__(self) -> int:
        return hash(self.kernels)

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, SumKernel):
            return False
        if len(self.kernels) != len(o.kernels):
            return False
        return all([k1 == k2 for (k1, k2) in zip(self.kernels, o.kernels)])

def gram(x1: Array, x2: Array, kernel: KernelLike) -> Array:
    """Computes the gram matrix for a kernel

    :param x1: [n x d]
    :param x2: [n x d]
    :return: [n x n]
    """
    assert x1.ndim == 2
    assert x2.ndim == 2

    return vmap(lambda x: vmap(lambda y: kernel(x, y))(x2))(x1)

###jax utils
T = TypeVar("T")


def tree_concatenate(trees: Iterable[T]) -> T:
    """Concatenates the leaves of a list of pytrees to produce a single pytree."""
    leaves, treedefs = zip(*[tree_util.tree_flatten(tree) for tree in trees])
    grouped_leaves = zip(*leaves)
    result_leaves = [jnp.concatenate(l) for l in grouped_leaves]
    return cast(T, treedefs[0].unflatten(result_leaves))


def batch_vmap(
    f: Callable[[KeyArray], T], rngs: KeyArray, batch_size: int, progress: bool = False
) -> T:
    """Equivalent to vmap(f)(rngs), but vmaps only batch_size rngs at a time.

    This reduces memory usage.

    :param progress: If True, displays a progress bar.
    """
    n_batches = int(ceil(rngs.shape[0] / batch_size))

    batch_results: list[T] = []
    if progress:
        iterator = tqdm(range(n_batches))
    else:
        iterator = range(n_batches)
    for batch_i in iterator:
        batch_rngs = rngs[batch_i * batch_size : (batch_i + 1) * batch_size]
        batch_results.append(vmap(f)(batch_rngs))

    return tree_concatenate(batch_results)


P = ParamSpec("P")


def to_scalar(f: Callable[P, Array]) -> Callable[P, Array]:
    @wraps(f)
    def f2(*args: P.args, **kwargs: P.kwargs) -> Array:
        return f(*args, **kwargs).reshape(())

    return f2


### bootstrapped_tests
BatchSize = int
ParallelMode = Union[Literal["all"], BatchSize]


@dataclass
class TestResult(Generic[P]):
    reject_null: bool
    theta_hat: P
    threshold: float
    test_statistic: float
    bootstrapped_test_stats: Optional[Array]
    bootstrapped_theta_hats: Optional[P]


class TestStatistic(ABC, Generic[P]):
    @abstractmethod
    def __call__(self, rng: KeyArray, theta_hat: P, ys: Array) -> Array:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


class WildTestStatistic(TestStatistic[P], Generic[P]):
    """Test statistic which supports the wild boostrap."""

    @abstractmethod
    def h_gram(
        self,
        rng: KeyArray,
        theta_hat: P,
        ys: Array,
    ) -> Array:
        pass


class Bootstrap(Enum):
    WILD = "wild"
    PARAMETRIC = "parametric"


def parametric_bootstrap_test(
    rng: KeyArray,
    ys: Array,
    estimator: Estimator[P],
    null_family: Type[SampleableDist[P, Array]],
    test_statistic: TestStatistic[P],
    n_bootstrap_samples: int,
    level: float = 0.05,
    save_null_distribution: bool = False,
    parallel_samples: ParallelMode = "all",
) -> TestResult[P]:
    rng, rng_input = jax.random.split(rng)
    theta_hat = estimator(rng_input, ys)

    # Resample a set of observations from the estimated distribution for each bootstrap
    # sampled. Rather than sampling n points during each bootstrap sample, just sample
    # n*n_bootstrap_samples points upfront, and split these.
    rng, rng_input = jax.random.split(rng)
    b_ys = null_family.sample_with_params(
        rng_input, theta_hat, n=ys.shape[0] * n_bootstrap_samples
    )
    b_ys = b_ys.reshape(n_bootstrap_samples, ys.shape[0], ys.shape[1])

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

    level = jnp.quantile(bootstrap_statistics, 1 - level).item()
    reject_null = statistic_value > level

    if not save_null_distribution:
        saved_boostrapped_theta_hats = None
        saved_bootstrapped_stats = None
    else:
        saved_boostrapped_theta_hats = bootstrap_theta_hats
        saved_bootstrapped_stats = bootstrap_statistics

    return TestResult(
        reject_null,
        theta_hat,
        level,
        statistic_value,
        saved_bootstrapped_stats,
        saved_boostrapped_theta_hats,
    )


def _make_parametric_bootstrap_samples(
    rng: KeyArray,
    b_ys: Array,
    estimator: Estimator[P],
    test_statistic: TestStatistic[P],
    n_bootstrap_samples: int,
    parallel_samples: ParallelMode,
) -> Tuple[P, Array]:
    if parallel_samples == "all":
        batch_size = n_bootstrap_samples
    else:
        batch_size = parallel_samples
    # To make things easier we insist that the total number of samples is a multiple of
    # the batch size, thus every batch can be the same size.
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
    estimator: Estimator[P],
    test_statistic: TestStatistic[P],
) -> Tuple[P, Array]:
    rng1, rng2 = jax.random.split(rng, num=2)
    b_ys = cast(Array, b_ys)
    b_theta_hat = estimator(rng1, b_ys)
    return b_theta_hat, test_statistic(rng2, b_theta_hat, b_ys)


def wild_bootstrap_test(
    rng: KeyArray,
    ys: Array,
    estimator: Estimator,
    test_statistic: WildTestStatistic,
    n_bootstrap_samples: int,
    level: float = 0.05,
    save_null_distribution: bool = False,
) -> TestResult:
    rng, rng_input = jax.random.split(rng)
    theta_hat = estimator(rng_input, ys)

    rng, rng_input = jax.random.split(rng)
    h_gram = test_statistic.h_gram(rng_input, theta_hat, ys)
    rng, rng_input = jax.random.split(rng)
    Bns, critical_value, statistic, reject_null = _compute_wild_boostrap_samples(
        rng_input, ys, h_gram, n_bootstrap_samples, level
    )

    saved_null_distribution = Bns if save_null_distribution else None
    return TestResult(
        reject_null.item(),
        theta_hat,
        critical_value.item(),
        statistic.item(),
        saved_null_distribution,
        bootstrapped_theta_hats=None,
    )


@partial(jit, static_argnames=("n_bootstrap_samples"))
def _compute_wild_boostrap_samples(
    rng: KeyArray,
    ys: Array,
    h_gram: Array,
    n_bootstrap_samples: int,
    level: float,
) -> Tuple[Array, Array, Array, Array]:
    n = ys.shape[0]
    rng, rng_input = jax.random.split(rng)
    Ws = jax.random.rademacher(rng_input, (n_bootstrap_samples, n))
    Bns = vmap(lambda W: (W @ h_gram @ W.T) / n**2)(Ws)

    critical_value = jnp.quantile(Bns, 1 - level)
    statistic = h_gram.mean()
    reject_null = statistic > critical_value

    return Bns, critical_value, statistic, reject_null


class KSDAnalyticEstimator(Estimator[Array]):
    def __init__(self, kernel: Kernel, dist_family: ExpFamilyDist) -> None:
        self.kernel = kernel
        self.dist_family = dist_family

    def __call__(self, rng: KeyArray, ys: Array) -> Array:
        return ksd_estimator_exp_family(self.kernel, self.dist_family, ys)

    @property
    def name(self) -> str:
        return "ksd"



class KSDStatistic(WildTestStatistic[P], Generic[P]):
    def __init__(self, kernel: Kernel, null: Type[UnnormalizedDist]) -> None:
        self.kernel = kernel
        self.null = null

    def __call__(self, rng: KeyArray, theta_hat: P, ys: Array) -> Array:
        return v_stat(self.kernel, self.null.score_with_params, theta_hat, ys)  #TODO change to v_stat for the wildbootstrap

    def h_gram(self, rng: KeyArray, theta_hat: P, ys: Array) -> Array:
        return h_gram(self.kernel, self.null.score_with_params, theta_hat, ys)

    @property
    def name(self) -> str:
        return f"ksd"


class ScoreFunc(Protocol):
    @staticmethod
    def __call__(params: Array, x: Array) -> Array:
        pass


class NaturalParameterInverse(Protocol):
    @staticmethod
    def __call__(eta_val: Array) -> Array:
        pass


class SufficientStatistic(Protocol):
    @staticmethod
    def __call__(x: Array) -> Array:
        pass


class B(Protocol):
    @staticmethod
    def __call__(x: Array) -> Array:
        pass


@partial(jit, static_argnames=("kernel", "score"))
def v_stat(kernel: Kernel, score: ScoreFunc, params: Array, ys: Array) -> Scalar:
    return h_gram(kernel, score, params, ys).mean()

@partial(jit, static_argnames=("kernel", "score"))
def u_stat(kernel: Kernel, score: ScoreFunc, params: Array, ys: Array) -> Scalar:
    n = ys.shape[0]
    H = h_gram(kernel, score, params, ys)  # (n, n)
    diagless_sum = H.sum() - jnp.trace(H)  # drop diagonal
    return diagless_sum / (n * (n - 1))



@partial(jit, static_argnames=("kernel", "score"))
def h_gram(kernel: Kernel, score: ScoreFunc, params: Array, ys: Array) -> Array:
    def h(y1: Array, y2: Array) -> Array:
        term1 = kernel(y1, y2) * score(params, y1) @ score(params, y2)
        term2 = score(params, y1) @ grad(kernel, argnums=1)(y1, y2)
        term3 = score(params, y2) @ grad(kernel, argnums=0)(y1, y2)
        term4 = jacfwd(jacrev(kernel, 0), 1)(y1, y2).trace()
        return term1 + term2 + term3 + term4

    return gram(ys, ys, h)


class Gaussian(SampleableAndNormalizedDist[Array, Array], ExpFamilyDist):
    def __init__(self, loc: Union[Scalar, float], scale: Union[Scalar, float]) -> None:
        self.loc = jnp.array(loc)
        self.scale = jnp.array(scale)

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
        return jnp.array([self.loc, self.scale])

    @staticmethod
    def natural_parameter(params: Array) -> Array:
        loc, scale = params[0], params[1]
        return jnp.array([loc / scale, -1 / (2 * scale)])

    @staticmethod
    def natural_parameter_inverse(eta_val: Array) -> Array:
        loc = -0.5 * eta_val[0] / eta_val[1]
        scale = jnp.sqrt(1 / (-2 * eta_val[1]))
        return jnp.array([loc, scale])

    @staticmethod
    def sufficient_statistic(x: Array) -> Array:
        return jnp.concatenate([x, x**2], axis=0)

    @staticmethod
    def b(x: Array) -> Array:
        # Note that the KSD only depends on db/dx. For Gaussians, b does not depend on
        # x, thus db/dx = 0. Thus, to simplify things, we just return zero here.
        return jnp.zeros(shape=())


def gaussian_fixed_scale(scale: Union[float, Scalar]):
    s = jnp.array(scale)

    class GaussianFixedScale(SampleableAndNormalizedDist[Array, Array], ExpFamilyDist):
        def __init__(self, loc: Scalar) -> None:
            self.loc = jnp.array([loc])

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

        def get_params(self) -> Array:
            return self.loc

        @staticmethod
        def natural_parameter(params: Array) -> Array:
            mean = params[0]
            return mean / s**2

        @staticmethod
        def natural_parameter_inverse(eta_val: Array) -> Array:
            mean = -0.5 * eta_val[0] / eta_val[1]
            std = jnp.sqrt(1 / (-2 * eta_val[1]))
            return jnp.array([mean, std])

        @staticmethod
        def sufficient_statistic(x: Array) -> Array:
            return jnp.concatenate([x, x**2], axis=0)

        @staticmethod
        def b(x: Array) -> Array:
            # Note that the KSD only depends on db/dx. For Gaussians, b does not depend on
            # x, thus db/dx = 0. Thus, to simplify things, we just return zero here.
            return jnp.zeros(shape=())

    return GaussianFixedScale


@partial(jit, static_argnames=("n"))
def _sample(rng: KeyArray, loc: Scalar, scale: Scalar, n: int) -> Array:
    return loc + scale * jax.random.normal(rng, shape=(n, 1))


def _score(x: Array, loc: Scalar, scale: Scalar) -> Array:
    return grad(to_scalar(norm.logpdf), argnums=0)(x.reshape(x.shape[0]), loc, scale)



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

    # big_lambda gram matrix has dimensions [n x n x k x k].
    # k is the dimension of eta.
    big_lambda_n = gram(ys, ys, Lambda).mean(0).mean(0)
    # Add a little bit to the diagonal to improve stability. Otherwise the matrix
    # inverse can explode.
    big_lambda_n = big_lambda_n + jnp.eye(big_lambda_n.shape[0]) * 1e-4

    # nu gram matrix has dimensions [n x n x k]
    nu_n = gram(ys, ys, nu)
    nu_n = nu_n.mean(0).mean(0)

    # We use linalg.solve rather than linalg.inv because it is more stable as it avoids
    # explicitly computing the matrix inverse.
    eta_estimate = jnp.linalg.solve(big_lambda_n, -0.5 * nu_n)

    theta_estimate = eta_inv(eta_estimate)
    return theta_estimate


#TODO list: 1. need h and grad_h wrt to theta evaluated at theta_hat or theta_star
#           2. U_n_f and  centered version of f wrt to the original sample X
#           3. bootstrap and T_b
#           4. need the KSD_test

# ===============================
# U_n f  and empirical centering
# ===============================

# U_n f = (1 / [n(n-1)]) * sum_{i != j} f(X_i, X_j),
# where {X_i}_{i=1}^n are the points in the ORIGINAL sample X.
@partial(jit, static_argnames=("f",))
def U_n_from_f(X: Array, f: Callable[[Array, Array], Array]) -> Array:
    """
    Computes U_n f for an arbitrary bivariate kernel f.
    Supports both scalar- and vector-valued outputs.

    Here X has shape (n, d) and denotes the ORIGINAL sample.
    """
    n = X.shape[0]
    F = vmap(lambda xi: vmap(lambda xj: f(xi, xj))(X))(X)  # (n,n) or (n,n,p)

    if F.ndim == 2:  # scalar-valued f
        mask = 1.0 - jnp.eye(n, dtype=F.dtype)
        return (F * mask).sum() / (n * (n - 1))
    else:            # vector-valued f with last dim p
        mask = 1.0 - jnp.eye(n, dtype=F.dtype)
        return (F * mask[..., None]).sum(axis=(0, 1)) / (n * (n - 1))


# f_n(x, y) = f(x, y)
#            - E_{X ~ Q_n}[ f(x, X) ]     = (1/n) * sum_{i=1}^n f(x, X_i),   X_i from ORIGINAL X
#            - E_{X ~ Q_n}[ f(X, y) ]     = (1/n) * sum_{i=1}^n f(X_i, y),   X_i from ORIGINAL X
#            + E_{X,X' ~ Q_n}[ f(X, X') ] = (1/n^2)* sum_{i=1}^n sum_{j=1}^n f(X_i, X_j), X_i,X'_j from ORIGINAL X
@partial(jit, static_argnames=("f",))
def f_empirically_centered_apply(
    x: Array,
    y: Array,
    X: Array,
    f: Callable[[Array, Array], Array],
) -> Array:
    """
    Evaluates the empirically centered core f_n at arbitrary (x, y),
    using the empirical measure Q_n built from the ORIGINAL sample X.
    """
    # E_{X ~ Q_n}[ f(x, X) ] = (1/n) * Σ_{i=1}^n f(x, X_i), with X_i from ORIGINAL X
    Ex_f_xX = vmap(lambda Xj: f(x, Xj))(X).mean(axis=0)

    # E_{X ~ Q_n}[ f(X, y) ] = (1/n) * Σ_{i=1}^n f(X_i, y), with X_i from ORIGINAL X
    Ex_f_Xy = vmap(lambda Xi: f(Xi, y))(X).mean(axis=0)

    # E_{X,X' ~ Q_n}[ f(X, X') ] = (1/n^2) * Σ_{i=1}^n Σ_{j=1}^n f(X_i, X_j), both from ORIGINAL X
    Exx_f_XX = vmap(lambda Xi: vmap(lambda Xj: f(Xi, Xj))(X))(X).mean(axis=(0, 1))

    # f_n(x, y)
    return f(x, y) - Ex_f_xX - Ex_f_Xy + Exx_f_XX


# ============================================================
# 1) Stein core h_θ and its parameter-gradient ∇_θ h_θ
# ============================================================

# h_θ(x,x′) = k(x,x′) s_θ(x)^⊤ s_θ(x′) + s_θ(x)^⊤ ∇_{x′}k(x,x′)
#           + s_θ(x′)^⊤ ∇_{x}k(x,x′) + tr[ ∇_{x}∇_{x′} k(x,x′) ]
@partial(jit, static_argnames=("kernel", "score"))
def h_scalar(kernel: Kernel, score: ScoreFunc, theta: Array, x: Array, xp: Array) -> Array:
    s_x  = score(theta, x)
    s_xp = score(theta, xp)
    k_x_xp = kernel(x, xp)
    dk_dxp = grad(kernel, argnums=1)(x, xp)
    dk_dx  = grad(kernel, argnums=0)(x, xp)
    # trace of the mixed Hessian wrt (x, x')
    hess_trace = jacfwd(jacrev(kernel, argnums=0), argnums=1)(x, xp).trace()
    return (s_x @ s_xp) * k_x_xp + (s_x @ dk_dxp) + (s_xp @ dk_dx) + hess_trace

# ∇_θ h_θ(x,x′)
@partial(jit, static_argnames=("kernel", "score"))
def grad_h_scalar(kernel: Kernel, score: ScoreFunc, theta: Array, x: Array, xp: Array) -> Array:
    return grad(lambda th: h_scalar(kernel, score, th, x, xp))(theta)


# ============================================================
# 2) Empirical centering on ORIGINAL X for a given θ
#    (matrix version used inside the bootstrap)
# ============================================================
@partial(jit, static_argnames=("kernel", "score"))
def centered_core_matrix_for_theta(kernel: Kernel, score: ScoreFunc, theta: Array, X: Array) -> Array:
    """
    Hc[i,j] = h_{θ,n}(X_i, X_j)
            = h_θ(X_i,X_j)
              - (1/n) Σ_m h_θ(X_i, X_m)
              - (1/n) Σ_m h_θ(X_m, X_j)
              + (1/n^2) Σ_{m,ℓ} h_θ(X_m, X_ℓ)
    with all expectations taken over the ORIGINAL sample X.
    Works for scalar h_θ; if you ever make it vector-valued, the broadcasting still fits.
    """
    H = h_gram(kernel, score, theta, X)  # (n,n)
    row_mean = H.mean(axis=1, keepdims=True)              # (n,1)
    col_mean = H.mean(axis=0, keepdims=True)              # (1,n)
    v_statistic = H.mean(axis=(0, 1), keepdims=True)  # shape (1,1)

    return H - row_mean - col_mean + v_statistic


# ============================================================
# 3) Composite ˜KSD bootstrap (U-statistic)
#    X = original sample; X_bootstrap = X[idx]
# ============================================================




# ============================================================
# 3) Composite ˜KSD bootstrap (U-statistic)
#    X = original sample; X_bootstrap = X[idx]
# ============================================================

def KSD_test(
    kernel: Kernel,
    score: ScoreFunc,
    rng: KeyArray,
    X: Array,                                 # original sample (n,d)
    B: int,
    level: float,
    estimator_fn: Callable[[KeyArray, Array], Array],  # (rng, sample) -> theta
):
    """
    Returns dict with keys: 'crit', 'T_obs', 'reject', 'theta_hat', 'draws'.

    Bootstrap draw T_b:
      1) idx ~ Multinomial w/ replacement over {1..n};  X_bootstrap = X[idx]
      2) θ*  = estimator_fn(rng, X_bootstrap)
      3) Build H^c(θ*) on ORIGINAL X, then take the submatrix [idx][:,idx]
      4) U_n over off-diagonal of that centered submatrix
      5) Gradient correction: (θ* - θ̂)ᵀ { U_n[∇_θ h_{θ*}] on X_bootstrap  -  U_n[∇_θ h_{θ̂}] on X }
    """
    n = X.shape[0]

    # θ̂ on original X  (done outside the jit so B never becomes a traced shape)
    rng, r_est = jax.random.split(rng)
    theta_hat = estimator_fn(r_est, X)

    # Precompute bootstrap keys on host; pass them into the jitted body
    keys_all = jax.random.split(rng, B)

    return _KSD_test_with_keys(kernel, score, X, B, level, estimator_fn, theta_hat, keys_all)


@partial(jit, static_argnames=("kernel", "score", "estimator_fn"))
def _KSD_test_with_keys(
    kernel: Kernel,
    score: ScoreFunc,
    X: Array,
    B: int,
    level: float,
    estimator_fn: Callable[[KeyArray, Array], Array],  # (rng, sample) -> theta
    theta_hat: Array,
    keys: Array,                                       # shape (B, 2)
):
    """
    Jitted inner: uses precomputed θ̂ and bootstrap keys.
    """
    n = X.shape[0]

    # Observed statistic T_obs = U_n h_θ̂ on ORIGINAL X
    T_obs = u_stat(kernel, score, theta_hat, X)

    # U_n[∇_θ h_{θ̂}] on ORIGINAL X (vector in R^p)
    grad_h_hat = tree_util.Partial(lambda x, y: grad_h_scalar(kernel, score, theta_hat, x, y))
    U_grad_orig = _u_offdiag_vec(X, grad_h_hat)  # (p,)

    def one_bootstrap(k: KeyArray) -> Array:
        k_idx, k_est = jax.random.split(k)
        idx = jax.random.choice(k_idx, n, shape=(n,), replace=True)
        X_bootstrap = X[idx]

        # θ* on bootstrap sample
        theta_star = estimator_fn(k_est, X_bootstrap)

        # H^c(θ*) built on ORIGINAL X, then restrict to bootstrap indices
        Hc_star = centered_core_matrix_for_theta(kernel, score, theta_star, X)
        Hc_sub = Hc_star[idx][:, idx]      #to have it evaluated at bootstrap samples, not original data
        # U_n off-diagonal mean on the submatrix
        U_star_c = (Hc_sub.sum() - jnp.trace(Hc_sub)) / (n * (n - 1))

        # Gradient correction
        grad_h_star = tree_util.Partial(lambda x, y: grad_h_scalar(kernel, score, theta_star, x, y))
        U_grad_star = _u_offdiag_vec(X_bootstrap, grad_h_star)  # (p,)
        corr = jnp.dot(theta_star - theta_hat, U_grad_star - U_grad_orig)

        return U_star_c + corr

    # Map over the provided B keys (no dynamic shape needed)
    T_b_vals = lax.map(one_bootstrap, keys)
    crit = jnp.quantile(T_b_vals, 1.0 - level)

    return {
        "crit": crit,
        "T_obs": T_obs,
        "reject": T_obs > crit,
        "theta_hat": theta_hat,
        "KSD-tildes": T_b_vals,
    }



# helper: U-mean over off-diagonal for vector-valued f(x,y) ∈ R^p
@partial(jit, static_argnames=("f_xy",))
def _u_offdiag_vec(X: Array, f_xy: Callable[[Array, Array], Array]) -> Array:
    F = vmap(lambda x: vmap(lambda y: f_xy(x, y))(X))(X)  # (n,n,p)
    n = X.shape[0]
    mask = 1.0 - jnp.eye(n, dtype=F.dtype)
    return (F * mask[..., None]).sum(axis=(0, 1)) / (n * (n - 1))


# ============================================================
# 4) Functional driver (no classes). Optional convenience.
# ============================================================
@partial(jit, static_argnames=("kernel", "null_family"))
def _estimator_adapter(kernel: Kernel, null_family: Type[UnnormalizedDist], rng: KeyArray, sample: Array) -> Array:
    # wraps your analytic estimator to (rng, sample) -> theta
    del rng  # not used by ksd_estimator_exp_family
    return ksd_estimator_exp_family(kernel, null_family, sample)

def run_ksd_test(
    rng: KeyArray,
    X: Array,
    kernel: Kernel,
    null_family: Type[UnnormalizedDist],
    *,
    B: int = 400,
    level: float = 0.05,
    mode: Literal["tilde_u"] = "tilde_u",
):
    """
     composite ~KSD bootstrap (U-stat).
    uses your analytic estimator under the hood.
    """
    estimator_fn = lambda r, samp: _estimator_adapter(kernel, null_family, r, samp)

    if mode == "tilde_u":
        return KSD_test(kernel, null_family.score_with_params, rng, X, B, level, estimator_fn)

# ================================
# Rejection-rate sanity experiment
# ================================

#fix such that it works for every scale
def median_lengthscale(X: jax.Array) -> float:
    n = X.shape[0]
    diffs = X[:, None, :] - X[None, :, :]
    d2 = (diffs ** 2).sum(-1)
    # take upper triangle distances (exclude diagonal)
    iu = jnp.triu_indices(n, k=1)
    med2 = jnp.median(d2[iu])
    # common 1D heuristic: l ≈ sqrt(0.5 * median pairwise squared distance)
    return float(jnp.sqrt(0.5 * med2 + 1e-12))



def rejection_rate_over_n_runs(
    *,
    mode: Literal["tilde_u", "wild_v"] = "tilde_u",
    n: int = 400,
    B: int = 100,
    level: float = 0.05,
    seed: int = 6,
    loc: float = 0.0,
    scale: float = 1.0,
    runs: int = 40,
    kernel: Kernel = GaussianKernel(l=1.0)
):

    """
    Runs the chosen KSD test 40 times on fresh samples from N(loc, scale),
    returns (#rejections, rate). Prints per-rep diagnostics.
    """
    key = jax.random.PRNGKey(seed)
    rejects = 0

    # Null distribution to generate data under H0^C
    null_dist = Gaussian(loc, scale)

    print(f"\n=== Running {runs} reps | mode={mode}, n={n}, B={B}, alpha={level} ===")
    for rep in range(runs):
        key, k_data, k_test = jax.random.split(key, 3)
        X = null_dist.sample(k_data, n)  # shape (n,1)


        res = run_ksd_test(
            rng=k_test,
            X=X,
            kernel=kernel,
            null_family=Gaussian,
            B=B,
            level=level,
            mode=mode,
        )

        # Bring small results to host first
        res = jax.device_get(res)  #TODO check if this helps

        # Now safe & cheap
        rejects += int(res["reject"])
        print(
            f"[rep {rep+1:02d}] reject={res['reject']}  "
            f"T_obs={res['T_obs']:+.6f}  crit={res['crit']:.6f}"
        )

    rate = rejects / runs
    print(f"\n==> Rejections: {rejects}/{runs} (rate = {rate:.3f})")
    return rejects, rate

# Example usage:
# 1) Composite ~KSD bootstrap (U-stat)
#rej_u, rate_u = rejection_rate_over_n_runs(mode="tilde_u")


# ============================================================
# Type-I error sweep + plots (one figure per fixed B)
# ============================================================
def plot_type1_vs_n_for_B_grid(
    n_values=(200, 300, 400, 500, 600),
    B_values=(100, 150, 200, 250),
    *,
    runs=50,
    level=0.05,
    seed=8,
    loc=0.0,
    scale=2.0,
    kernel:Kernel = GaussianKernel(l=1.0),
    save=False,        # set True to save PNGs
):

    for B in B_values:
        rates = []
        ses = []
        s = seed
        for n in n_values:
            # vary seed a bit across grid points for independence
            rej, rate = rejection_rate_over_n_runs(
                mode="tilde_u",
                n=n,
                B=B,
                level=level,
                seed=s,
                loc=loc,
                scale=scale,
                runs=runs,
                kernel=kernel,
            )
            rates.append(float(rate))
            # binomial standard error for the Monte Carlo estimate
            ses.append(math.sqrt(rate * (1.0 - rate) / runs))
            s += 1

        # --- make the figure for this fixed B ---
        plt.figure()
        plt.errorbar(n_values, rates, yerr=ses, fmt='-o', capsize=3)
        plt.axhline(level, linestyle='--')
        plt.xlabel('sample size n')
        plt.ylabel('Type I error (rejection rate)')
        plt.title(f'Type I error vs n  (B = {B}, runs = {runs})')
        plt.grid(True, alpha=0.3)
        if save:
            plt.savefig(f"type1_vs_n_B{B}.png", dpi=150, bbox_inches="tight")
        plt.show()

# === 1D: plot Type-I error for ~KSD (U-bootstrap) and Wild Bootstrap on the SAME samples ===
def plot_type1_vs_n_both_1d(
    n_values=(200, 300, 400, 500, 600),
    *,
    B: int = 100,
    runs: int = 100,
    level: float = 0.05,
    seed: int = 4,
    loc: float = 0.0,
    scale: float = 1.0,
    kernel: Kernel = GaussianKernel(l=1.0),
    save: bool = False,
):
    key = jax.random.PRNGKey(seed)

    # Estimators/statistics used by both procedures
    estimator_fn = lambda r, Y: ksd_estimator_exp_family(kernel, Gaussian, Y)  # for ~KSD test
    wild_estimator = KSDAnalyticEstimator(kernel, Gaussian)                     # for wild bootstrap
    wild_stat      = KSDStatistic(kernel, Gaussian)

    rates_tilde, ses_tilde = [], []
    rates_wild,  ses_wild  = [], []

    print(f"\n=== 1D | B={B} | runs={runs} | alpha={level} ===")
    for n in n_values:
        rej_tilde = 0
        rej_wild  = 0

        for _ in range(runs):
            key, k_data, k_bt1, k_bt2 = jax.random.split(key, 4)

            # Null data ~ N(loc, scale) — ONE sample used by BOTH tests
            X = Gaussian(loc, scale).sample(k_data, n)  # shape (n,1)

            # ~KSD (tilde-U) using your composite bootstrap
            res_t = KSD_test(kernel, Gaussian.score_with_params, k_bt1, X, B, level, estimator_fn)
            rej_tilde += int(res_t["reject"])

            # Wild bootstrap (V-statistic baseline)
            res_w = wild_bootstrap_test(k_bt2, X, wild_estimator, wild_stat, B, level)
            rej_wild += int(res_w.reject_null)

        # Monte Carlo rates + binomial SEs
        rate_t = rej_tilde / runs
        rate_w = rej_wild  / runs
        se_t   = math.sqrt(rate_t * (1.0 - rate_t) / runs)
        se_w   = math.sqrt(rate_w * (1.0 - rate_w) / runs)

        rates_tilde.append(rate_t); ses_tilde.append(se_t)
        rates_wild.append(rate_w);  ses_wild.append(se_w)

        print(f"n={n:3d} | ~KSD: {rate_t:0.3f} ± {se_t:0.3f} | Wild: {rate_w:0.3f} ± {se_w:0.3f}")

    # ---- one figure with both curves ----
    plt.figure()
    plt.plot(n_values, rates_tilde, marker='o', label='~KSD (U-bootstrap)')
    plt.plot(n_values, rates_wild, marker='s', label='Wild bootstrap (V-stat)')

    plt.axhline(level, linestyle='--')
    plt.xlabel('sample size n')
    plt.ylabel('Type I error (rejection rate)')
    plt.title(f'1D Gaussian: Type I error vs n (B={B}, runs={runs})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save:
        plt.savefig(f"type1_vs_n_1D_both_B{B}_runs{runs}.png", dpi=150, bbox_inches="tight")
    plt.show()


# ===========================
# Helpers: symmetric vectorization (vech) and its inverse
# ===========================
def _vech_indices(d: int):
    return jnp.tril_indices(d)

def vech_sym(M: Array) -> Array:
    i, j = _vech_indices(M.shape[0])
    return M[i, j]

def sym_from_vech(v: Array, d: int) -> Array:
    i, j = _vech_indices(d)
    M = jnp.zeros((d, d), dtype=v.dtype)
    M = M.at[i, j].set(v)
    M = M + M.T - jnp.diag(jnp.diag(M))
    return M

def _symmetrize(A: Array) -> Array:
    return 0.5 * (A + A.T)


# ===========================
# MVN (unknown mean & covariance) as an exponential family
# Cholesky parameterization: θ = [ μ (d), vec_tril(L_uncon) ],  Σ = L Lᵀ,
# where L = tril with positive diag via softplus(L_uncon[ii]).
# Natural params: η1 = Σ^{-1} μ,   η2 = -½ Σ^{-1}  (stored with vech)
# Sufficient stats: t(x) = [ x, vech(xxᵀ) ]
# ===========================
from jax.nn import softplus
from jax.scipy.linalg import solve_triangular

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
    # inverse of softplus for positive x
    return jnp.log(jnp.expm1(jnp.maximum(x, eps)))

def mvn_full_family_chol(d: int):
    pL = _tril_size(d)
    ii = jnp.diag_indices(d)

    def _unpack_params(theta: Array) -> Tuple[Array, Array]:
        mu = theta[:d]
        L_uncon = theta[d:]
        L = _unpack_tril(L_uncon, d)
        L = L.at[ii].set(softplus(L[ii]) + 1e-6)  # ensure SPD
        return mu, L

    class MVNFullChol(ExpFamilyDist, UnnormalizedDist[Array]):
        # ---- Score wrt x: s_θ(x) = Σ^{-1}(μ - x) implemented via triangular solves ----
        @staticmethod
        def score_with_params(params: Array, x: Array) -> Array:
            mu, L = _unpack_params(params)
            # Solve Σ s = (μ - x) using L Lᵀ s = (μ - x)
            b = (mu - x)
            y = solve_triangular(L, b, lower=True)           # L y = b
            s = solve_triangular(L.T, y, lower=False)        # Lᵀ s = y
            return s

        @staticmethod
        def unnorm_log_prob_with_params(params: Array, x: Array) -> Array:
            # not used by KSD; provide normalized log_prob instead if needed
            mu, L = _unpack_params(params)
            Sigma = L @ L.T + 1e-6 * jnp.eye(d)
            return multivariate_normal.logpdf(x, mean=mu, cov=Sigma)

        # Optional convenience
        @staticmethod
        def log_prob_with_params(params: Array, x: Array) -> Array:
            mu, L = _unpack_params(params)
            Sigma = L @ L.T + 1e-6 * jnp.eye(d)
            return multivariate_normal.logpdf(x, mean=mu, cov=Sigma)

        # ---- ExpFamilyDist API ----
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
            Sigma = L @ L.T + 1e-6 * jnp.eye(d)
            P = jnp.linalg.inv(Sigma)
            eta1 = P @ mu
            eta2 = -0.5 * P
            # store η2 with vech (symmetric)
            return jnp.concatenate([eta1, vech_sym(eta2)], axis=0)

        @staticmethod
        def natural_parameter_inverse(eta_val: Array) -> Array:
            eta1 = eta_val[:d]
            vech_eta2 = eta_val[d:]
            Eta2 = sym_from_vech(vech_eta2, d)
            Eta2 = 0.5 * (Eta2 + Eta2.T)
            P = -2.0 * Eta2 + 1e-6 * jnp.eye(d)   # ensure SPD
            Sigma = jnp.linalg.inv(P)
            L = jnp.linalg.cholesky(Sigma + 1e-6 * jnp.eye(d))
            mu = jnp.linalg.solve(P, eta1)
            # map back to unconstrained Cholesky params
            L_uncon = L.at[ii].set(_inv_softplus(L[ii]))
            theta = jnp.concatenate([mu, _pack_tril(L_uncon)], axis=0)
            return theta

        # ---- Sampling (used by experiments) ----
        @staticmethod
        def sample_with_params(rng: KeyArray, params: Array, n: int) -> Array:
            mu, L = _unpack_params(params)
            z = jax.random.normal(rng, shape=(n, d))
            return mu + z @ L.T

        # ---- Distribution base API ----
        def get_params(self) -> Array:  # optional, if ever instantiated
            raise NotImplementedError("Use static methods with explicit params.")

    return MVNFullChol


# ===========================
# Kernel helper (your form):  k(x,y) = exp(-||x-y||^2 / l),  l = d
# ===========================
def make_kernel_for_dim(d: int) -> GaussianKernel:
    return GaussianKernel(l=float(d))


# ===========================
# Type-I error curves for MVN (unknown μ, Σ), SAME samples for both tests
# H0: X ~ N(0, σ^2 I_d) with per-dimension σ specified via scale_by_d
# Shows ~KSD (U-bootstrap) and Wild bootstrap (baseline).
# ===========================
def plot_type1_vs_n_both_mvn(
    d_list=(1, 2, 3, 5, 10, 20),
    n_values=(100, 200, 300, 400, 500),
    *,
    B: int = 200,
    runs: int = 100,
    level: float = 0.05,
    seed: int = 11,
    scale_by_d: Union[float, dict[int, float]] = 1.0,
    save: bool = False,
):
    key = jax.random.PRNGKey(seed)

    for d in d_list:
        sigma = float(scale_by_d[d]) if isinstance(scale_by_d, dict) else float(scale_by_d)
        print(f"\n=== d={d} | B={B} | runs={runs} | α={level}  | H0 σ={sigma} ===")

        kernel = make_kernel_for_dim(d)
        MVN = mvn_full_family_chol(d)

        est_fn = lambda r, Y: ksd_estimator_exp_family(kernel, MVN, Y)   # ~KSD estimator
        wild_estimator = KSDAnalyticEstimator(kernel, MVN)               # for wild bootstrap
        wild_stat      = KSDStatistic(kernel, MVN)

        rates_tilde, rates_wild = [], []

        for n in n_values:
            rej_tilde = 0
            rej_wild  = 0

            for _ in range(runs):
                key, k_data, k_bt1, k_bt2 = jax.random.split(key, 4)

                # H0 data: N(0, σ^2 I_d)
                X = sigma * jax.random.normal(k_data, shape=(n, d))

                # ~KSD (composite U-bootstrap with empirical centering + gradient correction)
                res_t = KSD_test(kernel, MVN.score_with_params, k_bt1, X, B, level, est_fn)
                rej_tilde += int(res_t["reject"])

                # Wild bootstrap (V-stat baseline)
                res_w = wild_bootstrap_test(k_bt2, X, wild_estimator, wild_stat, B, level)
                rej_wild += int(res_w.reject_null)

            rate_t = rej_tilde / runs
            rate_w = rej_wild  / runs
            rates_tilde.append(rate_t)
            rates_wild.append(rate_w)

            print(f"n={n:4d} | ~KSD: {rate_t:0.3f} | Wild: {rate_w:0.3f}")

        plt.figure()
        plt.plot(n_values, rates_tilde, marker='o', linestyle='-', label='KSD test(U-bootstrap)')
        plt.plot(n_values, rates_wild, marker='s', linestyle='-', label='Wild bootstrap (V-stat)')

        # reference line (hidden from legend)
        plt.axhline(level, linestyle='--', color='0.5', linewidth=1, label='_nolegend_')

        plt.xlabel('sample size n')
        plt.ylabel('Type I error')
        plt.ylim(0, 0.1)  # <- force 0..0.1
        #plt.grid(False, alpha=0.3)
        plt.legend()
        plt.show()

# ===== example usage: per-dimension σ map with ℓ = d =====
scale_map = {10: 8.0, 20: 200.0}
plot_type1_vs_n_both_mvn(
    d_list=(10, 20),
    n_values=(200, 400, 600, 800),
    B=200,
    runs=500,
    level=0.05,
    seed=11,
    scale_by_d=scale_map,
    save=False,
)

# sigma= 6.0    600 0.04   800 0.028
# sigma = 12.0   200 0.02  400 0.024  600  0.044  800 0.032