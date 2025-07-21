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

import numpy as np  #TODO delete when not needed anymore


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
        return jnp.exp(-((x1 - x2) ** 2).sum() / (2 * self.l**2))


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
        return u_stat(self.kernel, self.null.score_with_params, theta_hat, ys) #TODO change it to u_stat

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
    diagless_sum = H.sum() - jnp.trace(H)  # drop diagonal without masking
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




####TEST
key = jax.random.PRNGKey(0)
n   = 300
mu  = 2.0
ys  = mu + jax.random.normal(key, (n,1))

kernel   = GaussianKernel(l=1.0)
null     = Gaussian  # the class you defined




#### KSD_test statistic

### generate gaussian r.v.
# True parameters
loc = 2.0
scale = 1.5
theta_true = jnp.array([loc, scale])

# Create dist object and sample
dist = Gaussian(loc, scale)
samples = dist.sample(key, n=300)  # shape: (300, 1)

estimator = KSDAnalyticEstimator(kernel, Gaussian)
theta_hat = estimator(key, samples)     #estimate theta

ksd_statistic = KSDStatistic(kernel, Gaussian)
ksd_value = ksd_statistic(key, theta_hat, samples) #estimate the KSD based on the samples

print("theta_hat:", theta_hat)
print("KSD U-statistic:", ksd_value)



def u_star_statistic(xs: Array, f: Callable[[Array, Array], Array]) -> Array:
    """
    Computes the U-statistic:
        U_n^* f = (1 / n(n - 1)) * sum_{i != j} f(x_i, x_j)

    Args:
        xs: Array of shape (n, d)
        f: Callable that takes two vectors (x_i, x_j) and returns scalar or vector

    Returns:
        Scalar or vector value of the U-statistic
    """
    n = xs.shape[0]

    # Compute the matrix of f(x_i, x_j)
    def apply_f_on_row(i):
        return vmap(lambda j: f(xs[i], xs[j]))(jnp.arange(n))

    f_matrix = vmap(apply_f_on_row)(jnp.arange(n))  # shape (n, n, p)

    # Remove diagonal and average the rest
    i_idx, j_idx = jnp.where(~jnp.eye(n, dtype=bool))
    sum_off_diag = f_matrix[i_idx, j_idx].sum(axis=0)

    #TODO for v statistic
    #sum_off_diag = f_matrix.sum(axis=(0,1))
    return sum_off_diag / (n * (n-1))  #TODO n*(n-1) for U statistic


def tilde_ksd(
    *,
    X: Array,              # original sample  (n,d)
    X_star: Array,         # bootstrap sample (n,d)
    n: int,
    theta_hat: Array,      # \hat{\theta}_n
    theta_star: Array,     # \theta*_n
    h_fn_star: Callable[[Array, Array], Array],
    grad_h_fn_star: Callable[[Array, Array], Array],
    grad_h_fn_hat: Callable[[Array, Array], Array],
) -> Array:
    """
    Computes the 'tilde' KSD^2 using distinct gradient evaluations
    at \theta* and \thetâ, as required in the bootstrap formula.
    """

    # Part 1: U*_n h_{\theta*}
    U_star_h = u_star_statistic(X_star, h_fn_star)

    # Part 2: gradients
    U_star_grad = u_star_statistic(X_star, grad_h_fn_star)   # (p,)
    U_orig_grad = u_star_statistic(X, grad_h_fn_hat)         # (p,)

    assert U_star_grad.ndim == 1, f"Expected 1D grad, got {U_star_grad.shape}"
    assert U_orig_grad.ndim == 1

    grad_term = jnp.dot(theta_star - theta_hat, U_star_grad - U_orig_grad)
    return n * U_star_h+ n * grad_term   #TODO redo the correction term n * grad_term


def h_fn_from_score(
    score_fn: Callable[[Array, Array], Array],
    kernel: Kernel
) -> Callable[[Array, Array, Array], Array]:
    """
    Returns a function h(theta, x, x') that implements the full Stein kernel as in your equation.

    Args:
        score_fn: (theta, x) -> score vector
        kernel:   a kernel object with call(x, x') -> scalar

    Returns:
        h(theta, x, x'): scalar value
    """
    def h(theta: Array, x: Array, xp: Array) -> Array:
        s_x  = score_fn(theta, x)
        s_xp = score_fn(theta, xp)

        k_x_xp = kernel(x, xp)
        grad_k_xp = grad(kernel, argnums=1)(x, xp)
        grad_k_x  = grad(kernel, argnums=0)(x, xp)
        hess_trace = jacfwd(jacfwd(kernel, 0), 1)(x, xp).trace()

        term1 = s_x @ s_xp * k_x_xp
        term2 = s_x @ grad_k_xp
        term3 = s_xp @ grad_k_x
        return term1 + term2 + term3 + hess_trace

    return h

# Set up score function
score_fn = Gaussian.score_with_params

# Create h_fn for both theta_star and theta_hat
h_fn = h_fn_from_score(score_fn, kernel)

# For gradient wrt theta: grad_h_fn(θ, x, x')
def grad_h_fn(theta: Array, x: Array, xp: Array) -> Array:
    return grad(lambda th: h_fn(th, x, xp))(theta)



def bootstrap_ksd_statistics(
    rng: KeyArray,
    ys: Array,
    *,
    B: int,
    estimator: KSDAnalyticEstimator,
    theta_hat: Array,
    kernel: Kernel,                  # still unused but kept
    null_class: type[UnnormalizedDist],
) -> Array:
    n = ys.shape[0]

    # freeze θ̂ for the whole run
    grad_h_fn_hat = partial(grad_h_fn, theta_hat)

    def one_bootstrap(key: KeyArray) -> Array:
        # 1. resample with replacement
        idx      = random.choice(key, 400, shape=(400,), replace=False)  #TODO back to n instead of 400
        ys_star  = ys[idx]

        # 2. re-estimate θ on resample
        theta_star = estimator(key, ys_star)

        # 3. build h and grad-h for θ*
        h_fn_star      = partial(h_fn,      theta_star)
        grad_h_fn_star = partial(grad_h_fn, theta_star)

        # 4. tilde-KSD statistic
        T_b = tilde_ksd(
            X               = ys,          # original sample
            X_star          = ys_star,     # bootstrap sample
            n               = n,
            theta_hat       = theta_hat,
            theta_star      = theta_star,
            h_fn_star       = h_fn_star,
            grad_h_fn_star  = grad_h_fn_star,
            grad_h_fn_hat   = grad_h_fn_hat,
        )
        return T_b

    keys   = random.split(rng, B)
    T_vals = vmap(one_bootstrap)(keys)   # shape (B,)
    return T_vals


def KSD_test(
    rng: KeyArray,
    X: Array,
    *,
    B: int,
    level: float = 0.05,
    kernel: Kernel,
    null_class: type[UnnormalizedDist],
    estimator: KSDAnalyticEstimator,
) -> dict:
    """
    Perform the KSD goodness-of-fit test using bootstrap.

    Args:
        rng: random key
        X: observed sample (n, d)
        B: number of bootstrap samples
        level: test significance level
        kernel: kernel object
        null_class: class of the null distribution (e.g., Gaussian)
        estimator: estimator object (e.g., KSDAnalyticEstimator)

    Returns:
        Dictionary with decision, threshold, test statistic, and bootstrap values.
    """
    n = X.shape[0]

    # Estimate theta_hat on original data
    theta_hat = estimator(rng, X)

    # Build h_fn and gradient functions
    score_fn = null_class.score_with_params
    h_fn_main = h_fn_from_score(score_fn, kernel)

    def grad_h_fn(theta, x, xp):
        return grad(lambda th: h_fn_main(th, x, xp))(theta)

    # Bootstrap statistics
    T_vals = bootstrap_ksd_statistics(
        rng=rng,
        ys=X,
        B=B,
        estimator=estimator,
        theta_hat=theta_hat,
        kernel=kernel,
        null_class=null_class,
    )

    # Compute observed test statistic
    ksd_statistic = KSDStatistic(kernel, null_class)
    test_stat = ksd_statistic(rng, theta_hat, X)*n

    # Compute bootstrap quantile threshold
    #threshold = jnp.quantile(T_vals, 1 - level)

    # Decision
    #reject = test_stat > threshold
    threshold = jnp.quantile(T_vals, 1 - level)
    reject = jnp.abs(test_stat) > threshold
    return {
        "reject_null": bool(reject),
        "test_statistic": float(test_stat),
        "threshold": float(threshold),
        "bootstrap_statistics": T_vals,
        "theta_hat": theta_hat,
    }



### TESTING
# Define test parameters
key = PRNGKey(22)
n = 1000
B = 20
loc = 2.0
scale = 1.5


### Our bootstrap
# Create null distribution and sample data
null_class = Gaussian   ###TODO if you want to use another density, implement it as a class, and use your sample data X
dist = null_class(loc, scale)
X = dist.sample(key, n=n)

# Define kernel and estimator
kernel = GaussianKernel(l=1.0)
estimator = KSDAnalyticEstimator(kernel, null_class)

# Run the KSD test
result = KSD_test(
    rng=key,
    X=X,
    B=B,
    level=0.05,
    kernel=kernel,
    null_class=null_class,
    estimator=estimator,
)

# Print results
print("==== KSD Test Result ====")
print(f"Reject H0: {result['reject_null']}")   #if reject null =false, model fits data well
print(f"Test statistic: {result['test_statistic']:.6f}")
print(f"Threshold: {result['threshold']:.6f}")
print(f"Theta_hat: {result['theta_hat']}")
print(f"First 5 bootstrap stats: {result['bootstrap_statistics'][:5]}")



### Wild bootstrap
ys = dist.sample(key, n)

# Define kernel and estimator
kernel = GaussianKernel(l=1.0)
estimator = KSDAnalyticEstimator(kernel, null_class)
stat_obj = KSDStatistic(kernel, null_class)

# Run wild bootstrap test
result = wild_bootstrap_test(
    rng=key,
    ys=ys,
    estimator=estimator,
    test_statistic=stat_obj,
    n_bootstrap_samples=300,  # Can be more
    level=0.05,
    save_null_distribution=True,
)

print("\n==== Wild Bootstrap KSD Test Result ====")
print(f"Reject H₀: {result.reject_null}")
print(f"Test statistic: {result.test_statistic:.6f}")
print(f"Threshold: {result.threshold:.6f}")
print(f"Theta_hat: {result.theta_hat}")
if result.bootstrapped_test_stats is not None:
    print(f"First 5 bootstrap stats: {result.bootstrapped_test_stats[:5]}")


### TESTING SIMULATION
# Configuration
key = jax.random.PRNGKey(0)
n_values = [10, 50, 100, 250, 500, 750, 1000]  # sample sizes
B = 100       # number of bootstrap samples
R = 200       # number of test repetitions
level = 0.05  # significance level

# Results placeholder
type1_wild = []

# Define test setup
kernel = GaussianKernel(l=1.0)
null_class = Gaussian
estimator = KSDAnalyticEstimator(kernel, null_class)
stat_obj = KSDStatistic(kernel, null_class)

# Main loop over sample sizes
for n in tqdm(n_values, desc="Estimating Type I error"):
    rejections = 0
    for r in range(R):
        key, subkey = jax.random.split(key)
        X = null_class(loc=2.0, scale=1.5).sample(subkey, n)

        key, test_key = jax.random.split(key)
        result = wild_bootstrap_test(
            rng=test_key,
            ys=X,
            estimator=estimator,
            test_statistic=stat_obj,
            n_bootstrap_samples=B,
            level=level,
        )

        if result.reject_null:
            rejections += 1

    type1_error = rejections / R
    type1_wild.append(type1_error)

# Plot Type I error
plt.figure(figsize=(6, 4))
plt.plot(n_values, type1_wild, 'gP-', label='wild bootstrap')
plt.axhline(y=level, linestyle='--', color='black', label='target level')
plt.xlabel("number of observations, n")
plt.ylabel("Type I error rate")
plt.title("Type I error under $H_0^C$ for the KSD test")
plt.ylim(0, 0.15)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#### our bootstrap type 1 error
# === Tilde-KSD Bootstrap Test ===
def tilde_ksd_test(
    rng, ys, B, estimator, test_statistic, level
):
    n = ys.shape[0]
    theta_hat = estimator(rng, ys)

    grad_h_fn_hat = jax.tree_util.Partial(grad_h_fn, theta_hat)

    def one_bootstrap(key):
        idx = jax.random.choice(key, n, shape=(n,), replace=True)  #TODO replace = True
        ys_star = ys[idx]

        #num_samples = ys.shape[0]
        #idx = jax.random.choice(key, num_samples, shape=(num_samples,), replace=False)
        #ys_star = ys[idx]
        theta_star = estimator(rng, ys_star)

        #key1, key2 = jax.random.split(key, 2)  # Split key inside
        #num_samples = 400
        #n_total = ys.shape[0]
        #idx = jax.random.choice(key1, n_total, shape=(num_samples,), replace=False)
        #ys_star = ys[idx]
        #theta_star = estimator(key2, ys_star)  # Use new key!

        h_fn_star = jax.tree_util.Partial(h_fn, theta_star)
        grad_h_fn_star = jax.tree_util.Partial(grad_h_fn, theta_star)

        T_b = tilde_ksd(
            X=ys,
            X_star=ys_star,
            n=n,
            theta_hat=theta_hat,
            theta_star=theta_star,
            h_fn_star=h_fn_star,
            grad_h_fn_star=grad_h_fn_star,
            grad_h_fn_hat=grad_h_fn_hat
        )
        return T_b,theta_star

    keys = jax.random.split(rng, B)
    T_b_vals=[]
    theta_star_vals=[]
    #T_b_vals,theta_star = jax.vmap(one_bootstrap)(keys)  #TODO use a for loop
    for i in keys:
        T_b,theta_star = one_bootstrap(i)
        T_b_vals.append(T_b)
        theta_star_vals.append(theta_star)


    T_obs = test_statistic(rng, theta_hat, ys) * n
    #threshold = jnp.quantile(T_b_vals, 1 - level)
    #reject = T_obs > threshold

    return theta_hat,theta_star_vals,T_obs,T_b_vals #reject#T_obs,T_b_vals#reject    #hier theta_star, theta ausgeben lassen  #TODO




# === Simulation of Type I Error ===
def simulate_type1_errors():
    n_values = [500]
    B = 10  #TODO test B=10 R=5
    R = 5
    level = 0.05
    kernel = GaussianKernel(l=1.0)
    null_class = Gaussian

    type1_boot = []
    type1_wild = []

    results = {
        "n": [],
        "theta_hat": [],
        "theta_star_vals": [],
        "T_obs": [],
        "T_b_vals": [],
    }


    master_key = jax.random.PRNGKey(0)

    for n in tqdm(n_values):
        rejects_boot = 0

        for _ in range(R):
            master_key, subkey1, subkey2 = jax.random.split(master_key, 3)

            dist = null_class(0, 0.5) #TODO changed this
            ys = dist.sample(subkey1, n=n)

            estimator = KSDAnalyticEstimator(kernel, null_class)
            test_statistic = KSDStatistic(kernel, null_class)

            if tilde_ksd_test(subkey2, ys, B, estimator, test_statistic, level):
                rejects_boot += 1

            theta_hat,theta_star_vals,T_obs,T_b_vals=tilde_ksd_test(subkey2, ys, B, estimator, test_statistic, level)
            # Store all outputs
            results["n"].append(n)
            results["theta_hat"].append(theta_hat)
            results["theta_star_vals"].append(theta_star_vals)
            results["T_obs"].append(T_obs)
            results["T_b_vals"].append(T_b_vals)

        type1_boot.append(rejects_boot / R)

    # Plot
   # plt.plot(n_values, type1_boot, 'r+-', label='tilde-KSD bootstrap')
    #plt.plot(n_values, type1_wild, 'gP-', label='wild bootstrap')
    #plt.axhline(y=level, color='black', linestyle='--', label='α = 0.05')
    plt.xlabel('Sample size (n)')
    plt.ylabel('Type I Error Rate')
    plt.title('Type I Error under $H_0$')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    return results

# === Run it ===
results=simulate_type1_errors()

def pretty_print_results(results, level=0.05, show=3):
    print("\n=== Tilde-KSD Bootstrap Test Summary ===")
    for i in range(len(results["n"])):
        n = results["n"][i]
        theta_hat = np.round(np.array(results["theta_hat"][i]), 4)
        T_obs = float(results["T_obs"][i])
        T_b_vals = [float(x) for x in results["T_b_vals"][i]]


        print(f"\n--- Run {i + 1} ---")
        print(f"Sample size n      : {n}")
        print(f"θ̂ (theta_hat)     : {theta_hat}")
        print(f"T_obs              : {T_obs:.4f}")

        # Show first few bootstrap stats
        print(f"T_b_vals (first {show}): {[round(val, 4) for val in T_b_vals[:show]]}")
        # Show first few theta_star
        theta_star_vals = [np.round(np.array(ts), 4) for ts in results["theta_star_vals"][i][:show]]
        print(f"θ*_star (first {show}): {theta_star_vals}")


pretty_print_results(results, level=0.05, show=10)

def mean_ksd_statistic_over_runs(
    rng: jax.Array,
    n: int,
    runs: int,
    kernel: Kernel,
    null_class: type[UnnormalizedDist],
) -> float:
    estimator = KSDAnalyticEstimator(kernel, null_class)
    ksd_vals = []

    for _ in range(runs):
        rng, subkey1, subkey2 = jax.random.split(rng, 3)
        dist = null_class(0.0, 0.5)
        ys = dist.sample(subkey1, n=n)
        theta_hat = estimator(subkey2, ys)
        ksd_val = v_stat(kernel, null_class.score_with_params, theta_hat, ys) * n
        ksd_vals.append(ksd_val)

    return jnp.mean(jnp.array(ksd_vals))

mean = mean_ksd_statistic_over_runs(
    rng=key,
    n=500,
    runs=1000,
    kernel=kernel,
    null_class=null_class
)
print(f"Mean KSD statistic over 1000 runs: {mean:.6f}")