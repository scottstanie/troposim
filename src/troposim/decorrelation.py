"""Simulate phase noise from decorrelation."""
import numpy as np
from tqdm import tqdm


def simulate(coherence=None, looks=1, nbins=200, rounding_threshold=0.05):
    """Simulate decorrelation noise from a given coherence.

    Parameters
    ----------
    coherence : int, ndarray
        Coherence magnitude of phase noise. (Default value = None)
    looks : int
        Number of looks. (Default value = 1)
    nbins : int, optional
        Granularity of phi bins, used for PDF generation. Defaults to 200.
    rounding_threshold : float, optional
        Threshold for rounding coherence to nearest multiple. Defaults to 0.05.
        Rounding occurs to speed up PDF calculations, since nearby coherence values
        produce nearly identical PDFs.

    Returns
    -------
    ndarray : Decorrelation phase noise. Shape is same as `coherence`

    Raises
    ------
    ValueError
        If `coherence` is not a scalar or array.

    References
    ----------
    Hanssen, 2001, Eq. 4.2.24
    (Derived by Barber, 1993, Lee et al., 1994, Joughin and Winebrenner, 1994)
    """
    out = np.zeros(coherence.shape)
    coh_rounded = _round_to(np.atleast_1d(coherence), rounding_threshold)
    if np.any(coh_rounded < 0) or np.any(coh_rounded > 1):
        raise ValueError("Coherence must be between 0 and 1")

    coh_rounded = np.clip(coh_rounded, 0.01, 0.99)
    looks = np.clip(looks, 1, 100)

    phi_bins = np.linspace(-np.pi, np.pi, nbins + 1, endpoint=True)
    # Will have a max of 100 unique coherence values
    unique_cohs, counts = np.unique(coh_rounded, return_counts=True)
    _, pdfs = phase_pdf(unique_cohs, looks, nbins=nbins, phi0=0.0)
    for pdf, coh, count in tqdm(zip(pdfs, unique_cohs, counts), total=len(unique_cohs)):
        samps = _sample_noise(phi_bins, pdf, size=count)
        idxs = coh_rounded == coh
        out[idxs] = samps
    return out


def phase_pdf(coherence, looks, nbins=200, phi0=0.0):
    """Compute the PDF of decorrleation phase noise for a given number of looks.
    
    Uses Eq. 4.2.24 from Hanssen, 2001, using the expression derived in
    Barber (1993), Lee et al. (1994), and Joughin and Winebrenner (1994)

    Parameters
    ----------
    coherence : int, ndarray
        Coherence magnitude of phase noise.
    looks : int
        Number of looks.
    nbins : int, optional
        Granularity of phi bins, used for PDF generation. Defaults to 200.
    phi0 : float, optional
        Center phase. Defaults to 0.0.

    Returns
    -------

    
    """
    from numpy import cos, pi
    from scipy import special as sc

    coherence = np.atleast_1d(coherence)
    phi = np.linspace(-pi, pi, nbins)
    # Add extra axis at end to make broadcasting work
    coherence = coherence[..., np.newaxis]
    # Keep adding extra axes to phi to match coherence.shape
    for ndim in [1, 2, 3]:
        if coherence.ndim > ndim:
            phi = phi[np.newaxis, :]

    phi -= phi0
    d = (1 - coherence**2) ** looks  # factor used twice
    num = sc.gamma(looks + 0.5) * d * np.abs(coherence) * cos(phi)
    z = coherence**2 * cos(phi) ** 2  # reused in hypergeometric function
    denom = 2 * np.sqrt(pi) * sc.gamma(looks) * (1 - z) ** (looks + 0.5)
    factor2 = d / 2 / pi * sc.hyp2f1(looks, 1, 0.5, z)
    return phi, num / denom + factor2


def _sample_noise(phi_bins, pdf, size=1):
    """Sample decorrelation phase noise for a given pdf, using `phase_pdf`.
    
    Length of phi_bins must be 1 greater than `len(pdf)`.

    Parameters
    ----------
    phi_bins :
        
    pdf : ndarray
        
    size : int or tuple of ints, optional
         (Default value = 1)

    Returns
    -------
    ndarray
    """
    from scipy import stats

    # phi, pdf = phase_pdf(coherence, looks, phi0, nbins)
    dist = stats.rv_histogram((pdf.ravel(), phi_bins))
    return dist.rvs(size=size)


def _round_to(x, step):
    """

    Parameters
    ----------
    x :
        
    step :
        

    Returns
    -------

    """
    return step * np.round(x / step)
