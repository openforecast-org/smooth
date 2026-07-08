"""
Distribution sampling functions for simulation-based prediction intervals.

These functions provide R-equivalent random samplers to ensure identical
results between R and Python implementations.
"""

import numpy as np
from scipy import stats


def rlaplace(n, mu=0, b=1, random_state=None):
    """
    Generate random samples from a Laplace distribution.

    Equivalent to R's rlaplace(n, mu, b).

    Parameters
    ----------
    n : int
        Number of samples to generate.
    mu : float
        Location parameter (mean).
    b : float
        Scale parameter.
    random_state : int or np.random.Generator, optional
        Random state for reproducibility.

    Returns
    -------
    np.ndarray
        Random samples from Laplace distribution.
    """
    if random_state is not None:
        if isinstance(random_state, int):
            rng = np.random.default_rng(random_state)
        else:
            rng = random_state
        return rng.laplace(mu, b, n)
    return np.random.laplace(mu, b, n)


def rs(n, mu=0, scale=1, random_state=None):
    """
    Generate random samples from an S-distribution.

    The S-distribution has PDF: f(x) = 1/(4*b^2) * exp(-|x-mu|/b) * |x-mu|/b
    Variance = 120 * scale^4

    Equivalent to R's rs(n, mu, scale).

    Parameters
    ----------
    n : int
        Number of samples to generate.
    mu : float
        Location parameter.
    scale : float
        Scale parameter.
    random_state : int or np.random.Generator, optional
        Random state for reproducibility.

    Returns
    -------
    np.ndarray
        Random samples from S-distribution.
    """
    if random_state is not None:
        if isinstance(random_state, int):
            rng = np.random.default_rng(random_state)
        else:
            rng = random_state
    else:
        rng = np.random.default_rng()

    # S-distribution can be generated as the difference of two Gamma(2, scale) variables
    # X = Gamma(2, scale) - Gamma(2, scale) + mu
    g1 = rng.gamma(2, scale, n)
    g2 = rng.gamma(2, scale, n)
    return g1 - g2 + mu


def rgnorm(n, mu=0, scale=1, shape=2, random_state=None):
    """
    Generate random samples from a Generalized Normal distribution.

    Equivalent to R's rgnorm(n, mu, scale, shape).

    Parameters
    ----------
    n : int
        Number of samples to generate.
    mu : float
        Location parameter.
    scale : float
        Scale parameter.
    shape : float
        Shape parameter (beta). When beta=2, this is the normal distribution.
    random_state : int or np.random.Generator, optional
        Random state for reproducibility.

    Returns
    -------
    np.ndarray
        Random samples from Generalized Normal distribution.
    """
    if random_state is not None:
        if isinstance(random_state, int):
            rng = np.random.default_rng(random_state)
        else:
            rng = random_state
    else:
        rng = np.random.default_rng()

    # Generate using the relationship with Gamma distribution
    # |X|^beta ~ Gamma(1/beta, 1)
    # X = sign * |X|
    gamma_samples = rng.gamma(1 / shape, 1, n)
    signs = rng.choice([-1, 1], n)
    return mu + signs * scale * (gamma_samples ** (1 / shape))


def ralaplace(n, mu=0, scale=1, alpha=0.5, random_state=None):
    """
    Generate random samples from an Asymmetric Laplace distribution.

    Equivalent to R's ralaplace(n, mu, scale, alpha).

    Parameters
    ----------
    n : int
        Number of samples to generate.
    mu : float
        Location parameter.
    scale : float
        Scale parameter.
    alpha : float
        Asymmetry parameter (0 < alpha < 1).
    random_state : int or np.random.Generator, optional
        Random state for reproducibility.

    Returns
    -------
    np.ndarray
        Random samples from Asymmetric Laplace distribution.
    """
    if random_state is not None:
        if isinstance(random_state, int):
            rng = np.random.default_rng(random_state)
        else:
            rng = random_state
    else:
        rng = np.random.default_rng()

    # Generate using exponential mixture
    u = rng.uniform(0, 1, n)
    e1 = rng.exponential(1, n)
    e2 = rng.exponential(1, n)

    # Asymmetric Laplace as mixture
    result = np.where(u < alpha, mu + scale * e1 / alpha, mu - scale * e2 / (1 - alpha))
    return result


def generate_errors(
    distribution,
    n,
    scale,
    obs_in_sample=None,
    n_param=None,
    shape=None,
    alpha=None,
    random_state=None,
):
    """
    Generate random errors for simulation based on distribution type.

    This function generates errors matching R's switch statement in forecast.adam().

    Parameters
    ----------
    distribution : str
        Distribution name (e.g., 'dnorm', 'dlaplace', 'dgamma').
    n : int
        Number of errors to generate.
    scale : float or np.ndarray
        Scale parameter (can be scalar or array for time-varying scale).
    obs_in_sample : int, optional
        Number of observations (needed for dt distribution).
    n_param : int, optional
        Number of parameters (needed for dt distribution).
    shape : float, optional
        Shape parameter (for dgnorm, dlgnorm).
    alpha : float, optional
        Asymmetry parameter (for dalaplace).
    random_state : int or np.random.Generator, optional
        Random state for reproducibility.

    Returns
    -------
    np.ndarray
        Array of random errors.
    """
    if random_state is not None:
        if isinstance(random_state, int):
            rng = np.random.default_rng(random_state)
        else:
            rng = random_state
    else:
        rng = np.random.default_rng()

    if distribution == "dnorm":
        return rng.normal(0, scale, n)

    elif distribution == "dlaplace":
        return rng.laplace(0, scale, n)

    elif distribution == "ds":
        return rs(n, 0, scale, random_state=rng)

    elif distribution == "dgnorm":
        return rgnorm(n, 0, scale, shape, random_state=rng)

    elif distribution == "dlogis":
        return rng.logistic(0, scale, n)

    elif distribution == "dt":
        if obs_in_sample is None or n_param is None:
            raise ValueError("obs_in_sample and n_param required for dt distribution")
        df = obs_in_sample - n_param
        return rng.standard_t(df, n) * scale

    elif distribution == "dalaplace":
        if alpha is None:
            raise ValueError("alpha required for dalaplace distribution")
        return ralaplace(n, 0, scale, alpha, random_state=rng)

    elif distribution == "dlnorm":
        # rlnorm(n, -scale^2/2, scale) - 1
        meanlog = -(scale**2) / 2
        return rng.lognormal(meanlog, scale, n) - 1

    elif distribution == "dinvgauss":
        # rinvgauss(n, 1, dispersion=scale) - 1
        # Using scipy for inverse Gaussian
        mu = 1
        lambda_param = 1 / scale  # dispersion = 1/lambda
        samples = stats.invgauss.rvs(
            mu / lambda_param, scale=lambda_param, size=n, random_state=rng
        )
        return samples - 1

    elif distribution == "dgamma":
        # rgamma(n, shape=scale^{-1}, scale=scale) - 1
        shape_param = 1 / scale
        return rng.gamma(shape_param, scale, n) - 1

    elif distribution == "dllaplace":
        # exp(rlaplace(n, 0, scale)) - 1
        return np.exp(rng.laplace(0, scale, n)) - 1

    elif distribution == "dls":
        # exp(rs(n, 0, scale)) - 1
        return np.exp(rs(n, 0, scale, random_state=rng)) - 1

    elif distribution == "dlgnorm":
        # exp(rgnorm(n, 0, scale, shape)) - 1
        if shape is None:
            raise ValueError("shape required for dlgnorm distribution")
        return np.exp(rgnorm(n, 0, scale, shape, random_state=rng)) - 1

    else:
        raise ValueError(f"Unknown distribution: {distribution}")


def normalize_errors(errors, error_type):
    """
    Normalize errors to have zero mean (additive) or unit mean multiplier
    (multiplicative).

    Applied for small simulation sample sizes (``nsim <= 500``) to remove
    Monte-Carlo bias in the simulated residual draws.

    Parameters
    ----------
    errors : np.ndarray
        Error matrix of shape (h, nsim).
    error_type : str
        'A' for additive, 'M' for multiplicative.

    Returns
    -------
    np.ndarray
        Normalized error matrix.
    """
    h, nsim = errors.shape

    if error_type == "A":
        # Center errors: subtract row means
        row_means = errors.mean(axis=1, keepdims=True)
        return errors - row_means
    else:
        # Normalize multipliers: (1+errors) / mean(1+errors) - 1
        multipliers = 1 + errors
        row_means = multipliers.mean(axis=1, keepdims=True)
        return multipliers / row_means - 1
