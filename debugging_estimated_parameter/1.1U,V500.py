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
    return diagless_sum / (n * (n - 1))  #TODO give out average of h(x,x)



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

# Define test parameters
key = jax.random.PRNGKey(0)
n = 1000
B = 20
loc = 0.0
scale = 0.5
# Create null distribution and sample data
null_class = Gaussian   ###TODO if you want to use another density, implement it as a class, and use your sample data X
dist = null_class(loc, scale)
# Define kernel and estimator
kernel = GaussianKernel(l=1.0)
estimator = KSDAnalyticEstimator(kernel, null_class)
test_statistic = KSDStatistic(kernel, null_class)


#### KSD_test statistic
def u_star_statistic(xs: Array, f: Callable[[Array, Array], Array], i_idx, j_idx) -> Array:
    n = xs.shape[0]

    def apply_f_on_row(i):
        return vmap(lambda j: f(xs[i], xs[j]))(jnp.arange(n))

    f_matrix = vmap(apply_f_on_row)(jnp.arange(n))  # shape (n, n, p)
    sum_off_diag = f_matrix[i_idx, j_idx].sum(axis=0)
    return sum_off_diag / (n * (n - 1))



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

    n = X.shape[0]
    i_idx, j_idx = np.where(~np.eye(n, dtype=bool))

    U_star_h = u_star_statistic(X_star, h_fn_star, i_idx, j_idx)
    U_star_grad = u_star_statistic(X_star, grad_h_fn_star, i_idx, j_idx)
    U_orig_grad = u_star_statistic(X, grad_h_fn_hat, i_idx, j_idx)

    assert U_star_grad.ndim == 1, f"Expected 1D grad, got {U_star_grad.shape}"
    assert U_orig_grad.ndim == 1

    grad_term = jnp.dot(theta_star - theta_hat, U_star_grad - U_orig_grad)
    return U_star_h + grad_term


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
        hess_trace = jacfwd(jacfwd(kernel, 0), 1)(x, xp).trace()  #TODO really important term, makes difference between 0 and 1 type 1 error

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

### Wild bootstrap
ys = dist.sample(key, n)



### TESTING SIMULATION WILDBOOTSTRAP
# Configuration
n_values = [250,500, 750, 1000]  # sample sizes
B = 200       # number of bootstrap samples
R = 200       # number of test repetitions
level = 0.05  # significance level

# Results placeholder
type1_wild = []
# Main loop over sample sizes
for n in tqdm(n_values, desc="Estimating Type I error"):
    rejections = 0
    for r in range(R):
        key, subkey = jax.random.split(key)
        X = null_class(loc=0.0, scale=1.5).sample(subkey, n)

        key, test_key = jax.random.split(key)
        result = wild_bootstrap_test(
            rng=test_key,
            ys=X,
            estimator=estimator,
            test_statistic=test_statistic,
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
    theta_hat = estimator(rng, ys)   #TODO gar nicht schaetzen sondern = echte parameter setzen

    grad_h_fn_hat = jax.tree_util.Partial(grad_h_fn, theta_hat)

    @jax.jit
    def one_bootstrap(key):
        idx = jax.random.choice(key, n, shape=(n,), replace=True)
        ys_star = ys[idx]
        theta_star = estimator(rng, ys_star)   #TODO theta_star=theta_hat
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
        return T_b

    keys = jax.random.split(rng, B)
    #T_b_vals = jax.vmap(one_bootstrap)(keys)
    T_b_vals = batch_vmap(one_bootstrap, keys, batch_size=5)  #TODO optimized version

    #T_b_vals = jnp.array(T_b_vals)
    T_obs = test_statistic(rng, theta_hat, ys)  #TODO delete the *n part
    threshold = jnp.quantile(T_b_vals, 1 - level)
    reject = T_obs > threshold

    return reject

# === Simulation of Type I Error ===
def simulate_type1_errors():
    n_values = [1000,2000]
    B = 200
    R = 100
    level = 0.05
    #kernel = GaussianKernel(l=1.0)
    #null_class = Gaussian

    type1_boot = []
    type1_wild = []

    master_key = jax.random.PRNGKey(0)
    dist = null_class(1, 2)  # TODO change this

    for n in tqdm(n_values):
        rejects_boot = 0

        for _ in range(R):
            master_key, subkey1, subkey2 = jax.random.split(master_key, 3)
            ys = dist.sample(subkey1, n=n)

            if tilde_ksd_test(subkey2, ys, B, estimator, test_statistic, level):
                rejects_boot += 1
        type1_boot.append(rejects_boot / R)

    # Plot
    plt.plot(n_values, type1_boot, 'r+-', label='tilde-KSD bootstrap')
    #plt.plot(n_values, type1_wild, 'gP-', label='wild bootstrap')
    plt.axhline(y=level, color='black', linestyle='--', label='α = 0.05')
    plt.xlabel('Sample size (n)')
    plt.ylabel('Type I Error Rate')
    plt.title('Type I Error under $H_0$')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

#simulate_type1_errors()

### 500 replications to get u and v statistics histograms and their mean
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
        dist = null_class(0.0, 10)
        ys = dist.sample(subkey1, n=n)
        theta_hat = estimator(subkey2, ys)
        ksd_val = u_stat(kernel, null_class.score_with_params, theta_hat, ys)
        ksd_vals.append(ksd_val)

    ksd_vals = jnp.array(ksd_vals)
    mean = jnp.mean(ksd_vals)
    std = jnp.std(ksd_vals)
    plt.figure(figsize=(8, 5))
    plt.hist(ksd_vals, bins=30, edgecolor='black', alpha=0.7)
    plt.axvline(mean, color='red', linestyle='--', label=f'Mean = {mean:.4f}')
    plt.axvline(mean + std, color='green', linestyle=':', label=f'+1 Std = {mean + std:.4f}')
    plt.axvline(mean - std, color='green', linestyle=':', label=f'-1 Std = {mean - std:.4f}')
    plt.title(f"KSD Statistic Histogram (n={n}, runs={runs}), Sigma of distribution=1.5")
    plt.xlabel("KSD value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return jnp.mean(jnp.array(ksd_vals)),jnp.std(jnp.array(ksd_vals))

mean,std = mean_ksd_statistic_over_runs(
    rng=key,
    n=10000,
    runs=100,
    kernel=kernel,
    null_class=null_class)
print(f"Mean KSD statistic over 500 runs: {mean:.6f}")
print(f"Standard Deviation: {std:.6f}")


def mean_uv_with_estimated_theta(
    rng: jax.Array,
    n: int,
    runs: int,
    kernel: Kernel,
    null_class: type[UnnormalizedDist],
    *,
    loc: float = 0.0,
    scale: float = 1.0,
    estimator: Optional[Estimator[jax.Array]] = None,
) -> Tuple[float, float, float, float]:
    """
    Repeats `runs` times:
      1) sample y_1..y_n ~ null_class(loc, scale)
      2) estimate θ̂ from the sample via `estimator` (defaults to KSDAnalyticEstimator)
      3) compute U- and V-statistics using θ̂
    Returns means & stds and plots histograms.
    """
    if estimator is None:
        estimator = KSDAnalyticEstimator(kernel, null_class)

    u_vals: list[float] = []
    v_vals: list[float] = []

    for _ in range(runs):
        rng, subkey1, subkey2 = jax.random.split(rng, 3)
        dist = null_class(loc, scale)
        ys = dist.sample(subkey1, n=n)                       # shape (n, d)

        theta_hat = estimator(subkey2, ys)                   # θ̂_n from current sample

        u_val = u_stat(kernel, null_class.score_with_params, theta_hat, ys)
        v_val = v_stat(kernel, null_class.score_with_params, theta_hat, ys)

        u_vals.append(float(u_val))
        v_vals.append(float(v_val))

    u_vals = jnp.array(u_vals)
    v_vals = jnp.array(v_vals)

    mean_u = float(jnp.mean(u_vals));  std_u = float(jnp.std(u_vals))
    mean_v = float(jnp.mean(v_vals));  std_v = float(jnp.std(v_vals))

    # --- Plot ---
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(u_vals, bins=30, edgecolor='black', alpha=0.8, density=True)
    plt.axvline(mean_u, linestyle='--', label=f'Mean = {mean_u:.4f}')
    plt.title(f"U-Stat (θ̂) Histogram (n={n}, runs={runs}, μ={loc}, σ={scale})")
    plt.xlabel("U (with θ̂)"); plt.ylabel("Density")
    plt.grid(True, alpha=0.3); plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(v_vals, bins=30, edgecolor='black', alpha=0.8, density=True)
    plt.axvline(mean_v, linestyle='--', label=f'Mean = {mean_v:.4f}')
    plt.title(f"V-Stat (θ̂) Histogram (n={n}, runs={runs}, μ={loc}, σ={scale})")
    plt.xlabel("V (with θ̂)"); plt.ylabel("Density")
    plt.grid(True, alpha=0.3); plt.legend()

    plt.tight_layout(); plt.show()

    return mean_u, std_u, mean_v, std_v

mean_u, std_u, mean_v, std_v = mean_uv_with_estimated_theta(
    rng=key,
    n=5000,
    runs=500,
    kernel=kernel,
    null_class=Gaussian,   # your `null_class`
    loc=0.0,
    scale=1.0,
    estimator=KSDAnalyticEstimator(kernel, Gaussian)
)

print(f"U-Stat (θ̂) mean over 500 runs: {mean_u:.6f} (std: {std_u:.6f})")
print(f"V-Stat (θ̂) mean over 500 runs: {mean_v:.6f} (std: {std_v:.6f})")

