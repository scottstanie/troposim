"""Simulate stratified atmospheric delay

Based mostly on Bekaert, 2015:
A spatially variable power law tropospheric correction technique for InSAR data
"""
import numpy as np


def simulate(dem, K_params={}, h0_params={}, alpha_params={}):
    """Simulate a set of stratified delay maps from a DEM
    Uses the power law scaling from Bekaert, 2015:
        `K * (h0 - h) ** alpha`

    Parameters
    ----------
    dem : ndarray
        DEM heights (in meters)
    K_params : dict
        parameters for sampling the atmospheric constant:
        options:
        'mean' (default = 0)
        'sigma': (default = 5e-6)
        'shape': (default = (1,))
        K is sampled from a laplacian distribution
        units: radians / meters^alpha
    h0_params : float
        top of atmosphere height.
        options:
        'mean' (default = 7000)
        'sigma': (default = 100)
        'shape': (default = (1,))
    alpha_params :
        parameters for sampling the power law scaling:
        options:
        'mean' (default = 1.4)
        'sigma': (default = 0.1)
        'shape': (default = (1,))

    Returns
    -------
    np.ndarray
    """
    K = sample_K(**K_params)
    alpha = sample_alpha(**alpha_params)
    h0 = sample_h0(**h0_params)
    return make_stratified_delay(dem, K, h0=h0, alpha=alpha)


def make_stratified_delay(dem, K, h0=7000, alpha=1.4, zero_mean=True):
    """Create a stratified delay map from a DEM

    Parameters
    ----------
    dem : ndarray
        DEM heights (in meters)
    K : float
        atmospheric constant: `K (h0 - h) ** alpha`
        units: radians / meters^alpha
    h0 : float
        top of atmosphere height. Defaults to 7000.
    alpha : float
        power law scaling. Defaults to 1.4.
    zero_mean :
        (Default value = True)

    Returns
    -------
    ndarray
    """
    K, h0, alpha = _get_param_sizes(K, h0, alpha)

    if np.any(h0 < dem.max()):
        raise ValueError(f"{h0 = } must be higher than DEM")
    dh = h0 - dem
    delay = K * dh**alpha
    if zero_mean:
        if delay.ndim == 3:
            delay -= delay.mean(axis=(1, 2), keepdims=True)
        else:
            delay -= delay.mean()
    return delay


def sample_K(mean=0.0, sigma=5e-6, shape=(1,)):
    """Sample a random K value from a normal distribution

    Parameters
    ----------
    mean :
        (Default value = 0.0)
    sigma :
        (Default value = 5e-6)
    shape : in or tuple[int]
        (Default value = (1,))

    Returns
    -------
    np.ndarray
    """
    return np.random.laplace(loc=mean, scale=sigma, size=shape)


def sample_alpha(mean=1.4, sigma=0.1, shape=(1,)):
    """Sample a random alpha value from a normal distribution

    Parameters
    ----------
    mean :
        (Default value = 1.4)
    sigma :
        (Default value = 0.1)
    shape :
        (Default value = (1, ))
    """
    return np.random.normal(loc=mean, scale=sigma, size=shape)


def sample_h0(mean=7000, sigma=100, shape=(1,)):
    """Sample a random h0 value from a normal distribution

    Parameters
    ----------
    mean :
        (Default value = 7000)
    sigma :
        (Default value = 100)
    shape :
        (Default value = (1, ))
    """
    return np.random.normal(loc=mean, scale=sigma, size=shape)


def _get_param_sizes(K, h0, alpha):
    if np.isscalar(K):
        K = np.array(K).reshape((1, 1, 1))
    else:
        K = np.array(K).reshape((len(K), 1, 1))
    if np.isscalar(h0):
        h0 = np.array(h0).reshape((1, 1, 1))
    else:
        h0 = np.array(h0).reshape((len(h0), 1, 1))
    if np.isscalar(alpha):
        alpha = np.array(alpha).reshape((1, 1, 1))
    else:
        alpha = np.array(alpha).reshape((len(alpha), 1, 1))
    nk, nh, na = len(K), len(h0), len(alpha)
    lengths = set([nk, nh, na])
    if len(lengths) > 2 or (len(lengths) > 1 and 1 not in lengths):
        raise ValueError("K, h0, and alpha must all have the same length or length 1")
    return K, h0, alpha
