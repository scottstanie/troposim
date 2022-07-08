"""Simulate phase noise from decorrelation."""
from collections import Counter

import numpy as np

try:
    from scipy import special as sc
    from scipy import stats
except ImportError:
    print("scipy not installed; needed for simulating decorrelation")


def phase_pdf(coherence, looks, phi0=0.0, nbins=200):
    """Compute the PDF of decorrleation phase noise for a given number of looks.

    Uses Eq. 4.2.24 from Hanssen, 2001
    """
    from numpy import cos, pi

    coherence = np.atleast_1d(coherence)
    # phi = np.linspace(0, 2 * np.pi, nbins)
    phi = np.linspace(-pi, pi, nbins)
    print(coherence.shape, phi.shape)
    # Add extra axis at end to make broadcasting work
    coherence = coherence[..., np.newaxis]
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
    """Sample decorrelation phase noise for a given number of looks.

    Uses Eq. 4.2.24 from Hanssen, 2001
    """
    # phi, pdf = phase_pdf(coherence, looks, phi0, nbins)
    dist = stats.rv_histogram((pdf.ravel(), phi_bins))
    return dist.rvs(size=size)


def simulate(coherence, looks, phi0=0.0, nbins=200):
    out = np.zeros(coherence.shape)
    coh_rounded = np.clip(np.atleast_1d(coherence).round(2), 0.01, 0.99)
    looks = np.clip(looks, 1, 100)

    phi_bins = np.linspace(-np.pi, np.pi, nbins + 1, endpoint=True)
    # Will have a max of 100 unique coherence values
    unique_cohs, counts = np.unique(coh_rounded, return_counts=True)
    # coh_counter = Counter(coh_rounded.ravel())
    _, pdfs = phase_pdf(unique_cohs, looks, phi0, nbins)
    for pdf, coh, count in zip(pdfs, unique_cohs, counts):
        samps = _sample_noise(phi_bins, pdf, size=count)
        idxs = coh_rounded == coh
        out[idxs] = samps
    return out


# psd1d = ndimage.mean(psd2d, r, index=np.arange(0, num_r))
