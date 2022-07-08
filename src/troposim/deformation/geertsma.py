import numpy as np
from scipy import integrate, special


def reservoir_subsidence(D, R, deltaH, nu=0.25, r=None, as_grid=False):
    """Compute the vertical subsidence according to cylindrical reservoir (Geertsma, 1973).

    Parameters
    ----------
    D : float
        Depth of the reservoir (m).
    R : float
        Radius of the reservoir (m).
    deltaH : float
        Change in reservoir height (m).
    nu : float
        Poisson's ratio. Defaults to 0.25.
    r : ndarray[float] optional
        Radius values at which to evaluate subsidence.
        Defaults to 100 points spaced equally between 0 and R.
    as_grid : bool
        Return the subsidence as a square grid. Defaults to False.
        If False, returns two 1d arrays

    Returns
    -------

    
    """
    # We don't need biot's coefficient if we input the DeltaH.
    # This is related to the pressure drop through DeltaH = H * c_m * DeltaP
    # Notes on other way's to get biot coefficient
    # G_fr = E / (2 * (1 + nu))  # Frame Shear Modulus
    # K_fr = (2 / 3) * G_fr * (1 + nu) / (1 - 2 * nu)  # Frame Bulk Modulus
    # Cm = (1 + nu) * (1 - 2 * nu) / (E * (1 - nu))  # Uniaxial Compressibility
    # alpha = 1 - K_fr / K_mineral  # Biot Coefficient

    # Source: https://github.com/Abakumov/MLIB (though this is wrong)
    def f_A(a, rho):
        """

        Parameters
        ----------
        a :
            
        rho :
            

        Returns
        -------

        
        """
        return np.exp(-D * a) * special.j1(a * R) * special.j0(a * rho)

    if r is None:
        r = np.linspace(0, 2 * R, 100)

    if np.isscalar(r):
        r = np.array([r])

    Uz = np.zeros(r.shape)

    # for i in range(len(rho)):
    # for idx, cur_rho in enumerate(rho):
    for idx, cur_r in enumerate(r):
        int_val = integrate.quad(lambda a: f_A(a, cur_r), 0, np.inf)[0]
        Uz[idx] = 2 * (1 - nu) * deltaH * R * int_val

    if as_grid:
        return _make_square_grid(Uz, r)
    else:
        return Uz, r


def _make_square_grid(Uz, r):
    """

    Parameters
    ----------
    Uz :
        
    r :
        

    Returns
    -------

    
    """
    # Uz, r = reservoir_subsidence(D, R, deltaH, nu=nu, r=r)
    n = len(r)

    # Make the corners of the square equal to R
    xvals = np.linspace(-1 / np.sqrt(2), 1 / np.sqrt(2), n) * max(r)
    XX, YY = np.meshgrid(xvals, xvals)
    RR = np.sqrt(XX ** 2 + YY ** 2)

    r_idxs = find_closest_idxs(r, RR)
    Uz_square = Uz[r_idxs]
    return Uz_square, RR


def plot(D, R, deltaH, nu=0.25, r=None):
    """

    Parameters
    ----------
    D :
        
    R :
        
    deltaH :
        
    nu :
        (Default value = 0.25)
    r :
        (Default value = None)

    Returns
    -------

    
    """
    import matplotlib.pyplot as plt

    Uz_square, RR = reservoir_subsidence(D, R, deltaH, nu=nu, r=r, as_grid=True)
    xmin, xmax = np.array([RR.min(), RR.max()]) / np.sqrt(2)
    fig, ax = plt.subplots()
    axim = ax.imshow(-Uz_square, extent=[xmin, xmax, xmin, xmax], origin="lower")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Vertical subsidence (m)")
    fig.colorbar(axim)
    return Uz_square, RR


def find_closest_idxs(known_array, test_array):
    """

    Parameters
    ----------
    known_array :
        
    test_array :
        

    Returns
    -------

    
    """
    index_sorted = np.argsort(known_array)
    known_array_sorted = known_array[index_sorted]
    known_array_middles = (
        known_array_sorted[1:] - np.diff(known_array_sorted.astype("f")) / 2
    )
    idx1 = np.searchsorted(known_array_middles, test_array)
    indices = index_sorted[idx1]
    return indices
