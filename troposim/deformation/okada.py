"""Generates surface deformation from fault slip according to Okada's 
rectangular plane model.

Requres okada_wrapper:
https://github.com/tbenthompson/okada_wrapper/
"""
import numpy as np
from scipy.ndimage import rotate

try:
    from okada_wrapper import dc3dwrapper
except ImportError:
    print("okada_wrapper not found. See for instructions:")
    print(" https://github.com/tbenthompson/okada_wrapper/")
    print("Or, pip install okada_wrapper")


def displacement(
    source_depth,
    dip_angle,
    strike_width,
    dip_width,
    alpha=None,
    mu=None,
    lam=None,
    poisson_ratio=0.25,
    strike_angle=0.0,
    rake_angle=0.0,
    slip_magnitude=0.0,
    strike_slip=0.0,
    dip_slip=0.0,
    obs_depth=0.0,
    xy_extent=None,
    shape=None,
    n=200,
):
    """source_depth (float): the depth of the fault origin
    dip_angle (float): the dip-angle, in degrees, of the rectangular dislocation surface
    strike_width (float):  the along-strike width of the surface
    dip_width (float):  the along-dip range of the surface
    alpha = (lambda + mu) / (lambda + 2 * mu)
    dip_slip (float): the dip-slip, in meters, of the rectangular dislocation surface
    strike_slip (float): the strike-slip, in meters, of the rectangular dislocation surface
    rake_angle (float): alternnative to strike_slip and dip_slip, the rake angle of the fault.
        See here for conversions of rake angle to strike_slip and dip_slip:
        https://croninprojects.org/Vince/Course/IntroStructGeol/Pitch.pdf
    slip_magnitude (float): the magnitude of the fault slip, in meters. For use with rake_angle
    obs_depth (float): the depth of the observation point. Defaults to 0.0, the surface
    n (int): the number of points per direction to sample the surface. Defaults to 200.

    Parameters
    ----------
    source_depth :
        
    dip_angle :
        
    strike_width :
        
    dip_width :
        
    alpha :
        (Default value = None)
    mu :
        (Default value = None)
    lam :
        (Default value = None)
    poisson_ratio :
        (Default value = 0.25)
    strike_angle :
        (Default value = 0.0)
    rake_angle :
        (Default value = 0.0)
    slip_magnitude :
        (Default value = 0.0)
    strike_slip :
        (Default value = 0.0)
    dip_slip :
        (Default value = 0.0)
    obs_depth :
        (Default value = 0.0)
    xy_extent :
        (Default value = None)
    shape :
        (Default value = None)
    n :
        (Default value = 200)

    Returns
    -------
    u_out : ndarray
        the displacement field
        shape: (3, ny, nx)
        Each pixel is [ux, uy, uz] 
    xx : ndarray
        the x-coordinates of the displacement field
    yy : ndarray
        the y-coordinates of the displacement field
    
    Examples
    --------
    >>> from okada_wrapper import dc3dwrapper
    >>> success, u, grad_u = dc3dwrapper(0.6, [1.0, 1.0, -1.0],\
                                3.0, 90, [-0.7, 0.7], [-0.7, 0.7],\
                                [1.0, 0.0, 0.0])
    """
    if alpha is None:
        if lam is None:
            if mu is None:
                raise ValueError("Must specify alpha or lam or mu")
            lam = 2 * mu * poisson_ratio / (1 - 2 * poisson_ratio)
        alpha = (mu + lam) / (mu + 2 * lam)

    strike_range = [-strike_width / 2, strike_width / 2]
    dip_range = [-dip_width / 2, dip_width / 2]

    if rake_angle is not None:
        strike_slip, dip_slip = rake_to_slips(rake_angle, slip_magnitude)
    # 3rd is "opening" of fault
    slip_vector = np.array([strike_slip, dip_slip, 0.0])

    if xy_extent is None:
        xy_extent = [-strike_width, -strike_width, strike_width, strike_width]
    left, bot, right, top = xy_extent
    if shape is not None:
        ny, nx = shape
    else:
        ny = nx = n
    xx, yy = np.linspace(left, right, ny), np.linspace(bot, top, nx)

    u_out = np.zeros((3, n, n))
    for j, x in enumerate(xx.ravel()):
        for i, y in enumerate(yy.ravel()):
            success, u, grad_u = dc3dwrapper(
                alpha,
                [x, y, obs_depth],
                source_depth,
                dip_angle,
                strike_range,
                dip_range,
                slip_vector,
            )
            if success != 0:
                raise ValueError(f"dc3dwrapper failed: {success}")
            u_out[:, i, j] = u

    if strike_angle:
        # Also, idk if it's the right north-angle convention.
        # rotates clockwise when plotted with pcolormesh X/Y
        # TODO: make this -90 by default? So it's north up? (assuming y is north)
        u_out = rotate(
            u_out, strike_angle, axes=(-1, -2), reshape=False, mode="nearest"
        )
    return u_out, xx, yy


def project_to_los(u, los):
    """Projects a displacement field to a line of sight.

    Parameters
    ----------
    u : ndarray
        the displacement field from `displacement`
        shape: (3, ny, nx): [ux, uy, uz]
    los : ndarray
        the line of sight unit vector at the observation point
        Can be a single vector, or an array the same size as u.
        shape: (3,): [losx, losy, losz]
        Can be the "ENU" vector of coefficients.

    Returns
    -------
    u_out : 2D ndarray
        the projected displacement field
    """
    los = np.array(los)
    # assert this?
    # los = los / np.linalg.norm(los)
    if los.shape == (3,):
        los = los.reshape(3, 1, 1)
    elif los.shape != u.shape:
        raise ValueError("los must be a 3-vector or the same size as u")
    u_los = np.sum(u * los, axis=0)
    return u_los


def rake_to_slips(rake_angle, slip_magnitude):
    """Convert a rake angle and magnitude to strike-slip and dip-slip components.

    Parameters
    ----------
    rake_angle : float
        the rake angle of the fault, in degrees
        
    slip_magnitude : float
        the magnitude of the fault slip, in meters. For use with rake_angle

    Returns
    -------
    strike_slip : float
        the strike-slip, in meters, of the rectangular dislocation surface
    dip_slip : float
        the dip-slip, in meters, of the rectangular dislocation surface

    
    Examples
    --------
    >>> tuple(np.round(rake_to_slips(-90, 0.1), 4)) # purely dip slip on normal fault
    (0.0, -0.1)
    >>> tuple(np.round(rake_to_slips(0, 0.1), 4)) # pure left lateral strike slip
    (0.1, 0.0)
    >>> tuple(np.round(rake_to_slips(90, 0.1), 4)) # reverse fault slip
    (0.0, 0.1)
    """
    strike_slip = slip_magnitude * np.cos(np.deg2rad(rake_angle))
    dip_slip = slip_magnitude * np.sin(np.deg2rad(rake_angle))
    return strike_slip, dip_slip


def random_displacement(
    source_depth_range=(0.3e3, 5e3),
    dip_angle_range=(0, 90),
    strike_angle_range=(0, 360),
    strike_width_range=(5e3, 20e3),
    dip_width_range=(300, 1000),
    rake_angle_range=(-89, -91),  # dip slip
    slip_magnitude_range=(0.05, 0.5),
    alpha=0.66,
    resolution=180,
    shape=(200, 200),
    seed=None,
    **kwargs,
):
    """

    Parameters
    ----------
    source_depth_range : tuple[float, float]
        (Default value = (0.3e3,5e3)
    dip_angle_range : tuple[float, float]
        (Default value = (0, 90)
    strike_angle_range : tuple[float, float]
        (Default value = (0, 360)
    strike_width_range : tuple[float, float]
        (Default value = (5e3, 20e3)
    dip_width_range : tuple[float, float]
        (Default value = (300, 1000)
    rake_angle_range : tuple[float, float]
        (Default value = (-89, -91)
    slip_magnitude_range : tuple[float, float]
        (Default value = (0.05, 0.5)
    alpha : float
        (Default value = 0.66)
    resolution : float
        (Default value = 180)
    shape : tuple[int, int]
        (Default value = (200, 200)
        
    seed : int
        (Default value = None)

    Returns
    -------
    ndarray
    """
    # see https://stackoverflow.com/a/49849045/4174466 for why RandomState
    rng = np.random.default_rng(seed=seed)
    kwargs["source_depth"] = rng.uniform(*source_depth_range)
    kwargs["dip_angle"] = rng.uniform(*dip_angle_range)
    kwargs["strike_angle"] = rng.uniform(*strike_angle_range)
    kwargs["strike_width"] = rng.uniform(*strike_width_range)
    kwargs["dip_width"] = rng.uniform(*dip_width_range)
    kwargs["rake_angle"] = rng.uniform(*rake_angle_range)
    kwargs["slip_magnitude"] = rng.uniform(*slip_magnitude_range)
    # print(kwargs)

    kwargs["alpha"] = alpha
    ny, nx = shape
    xy_extent = [
        -nx / 2 * resolution,
        -ny / 2 * resolution,
        nx / 2 * resolution,
        ny / 2 * resolution,
    ]
    kwargs["shape"] = shape
    kwargs["xy_extent"] = xy_extent
    return displacement(**kwargs)


def test_okada(
    dip_slip=0.1,
    strike_slip=0.0,
    dip_angle=45.0,
    rake_angle=None,
    slip_magnitude=0.0,
    strike_angle=0.0,
):
    """

    Parameters
    ----------
    dip_slip :
        (Default value = 0.1)
    strike_slip :
        (Default value = 0.0)
    dip_angle :
        (Default value = 45.0)
    rake_angle :
        (Default value = None)
    slip_magnitude :
        (Default value = 0.0)
    strike_angle :
        (Default value = 0.0)

    Returns
    -------
    u : ndarray
        the displacement field from `displacement`
    xx : ndarray
        the x-coordinates of the displacement field
    yy : ndarray
        the y-coordinates of the displacement field
    """
    import matplotlib.pyplot as plt

    # Mentone Eq:
    # strike: 082°, dip: 37°, and rake: −109°. source_depth=5km, 14km along strike
    # Pecos modeling: dip: 57, rake: -90 (all dip-slip), source_depth: .9 km,
    #  15 km along strike, dip_width: .6 km
    # TODO: conversion from rake angle to dip/strike slip vector

    mu = lam = 10e9  # Pa, same as Young's = 25e9 and poisson's = 0.25
    alpha = (lam + mu) / (lam + 2 * mu)
    # In [103]: E, nu = 25e9, 0.25
    # In [106]: sp.solve([mu * (3*lam + 2*mu) / (lam + mu) - E, lam/2/(lam+mu) - nu])
    # Out[106]: [{lam: 10000000000.0000, mu: 10000000000.0000}]
    depth = 900  # meters
    strike_width = 14e3  # meters
    dip_width = 600  # meters

    # dip_slip = 0.1  # meters
    # strike_slip = 0.0  # meters

    u, xx, yy = displacement(
        depth,
        dip_angle,
        strike_width,
        dip_width,
        alpha=alpha,
        dip_slip=dip_slip,
        strike_slip=strike_slip,
        rake_angle=rake_angle,
        slip_magnitude=slip_magnitude,
        strike_angle=strike_angle,
    )

    uz = u[-1]
    vm = np.abs(uz).max()
    plt.figure()
    plt.pcolormesh(xx, yy, u[-1], vmax=vm, vmin=-vm, cmap="RdBu")
    # plt.imshow(u[-1], vmax=vm, vmin=-vm, cmap="RdBu")
    plt.colorbar()
    return u, xx, yy
