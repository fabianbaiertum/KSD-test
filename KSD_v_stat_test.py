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
        return v_stat(self.kernel, self.null.score_with_params, theta_hat, ys) #TODO change it to u_stat

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

@partial(jit, static_argnames=("kernel", "score"))  #TODO use this instead of the V-statistic
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

n = 300
R = 500
kernel = GaussianKernel(l=1.0)
null_dist = Gaussian(loc=0.0, scale=1.0)
dist_class = type(null_dist)

ksd_values = []
rng = jax.random.PRNGKey(42)

for i in tqdm(range(R)):
    rng, key = jax.random.split(rng)
    ys = null_dist.sample(key, n)
    theta_hat = ksd_estimator_exp_family(kernel, dist_class, ys)
    Tn = v_stat(kernel, dist_class.score_with_params, theta_hat, ys)
    ksd_values.append((Tn * n).item())

ksd_values = jnp.array(ksd_values)

# Plotting
plt.figure(figsize=(8, 4))
plt.hist(ksd_values, bins=40, density=True, alpha=0.75, edgecolor="black")
plt.title("Empirical Distribution of n * KSD (V-statistic)")
plt.xlabel("n * KSD")
plt.ylabel("Density")
plt.grid(True)
plt.tight_layout()
plt.show()


### with true theta  #TODO true theta scaled by n
n = 300
R = 500
kernel = GaussianKernel(l=1.0)
true_theta = jnp.array([0.0, 1.0])  # True parameters of Gaussian
null_dist = Gaussian(loc=true_theta[0], scale=true_theta[1])

ksd_values_true = []
rng = jax.random.PRNGKey(43)  # different seed to distinguish from estimated theta case

for i in tqdm(range(R)):
    rng, key = jax.random.split(rng)
    ys = null_dist.sample(key, n)
    Tn = v_stat(kernel, null_dist.score_with_params, true_theta, ys)
    ksd_values_true.append((Tn * n).item())

ksd_values_true = jnp.array(ksd_values_true)

# Plotting
plt.figure(figsize=(8, 4))
plt.hist(ksd_values_true, bins=40, density=True, alpha=0.75, edgecolor="black")
plt.title("Empirical Distribution of n * KSD (U-statistic, true θ)")
plt.xlabel("n * KSD")
plt.ylabel("Density")
plt.grid(True)
plt.tight_layout()
plt.show()
