import numpy as np
import pytest
from numpy.polynomial import Polynomial
from numpy.testing import assert_allclose, assert_array_equal

from troposim import turbulence


def test_default():
    turbulence.simulate()


def test_shapes():
    freq0 = 2.0e-3
    out = turbulence.simulate(shape=(10, 10), freq0=freq0)
    assert out.shape == (10, 10)
    out = turbulence.simulate(shape=(2, 10, 10), freq0=freq0)
    assert out.shape == (2, 10, 10)


def test_get_psd():
    """ """
    shape = (200, 200)
    p0 = 1e-3
    b = -2.5
    freq0 = 2.0e-3
    out = turbulence.simulate(shape=shape, beta=b, freq0=freq0, p0=p0)
    psd = turbulence.Psd.from_image(out, freq0=freq0, deg=1)

    beta_tol = 0.15
    assert abs(psd.beta.item().coef[1] - b) < beta_tol
    p0_tol = 0.05
    assert abs(psd.p0 - p0) < p0_tol


def _beta_is_valid(beta, num_images):
    beta = turbulence._standardize_beta(beta, num_images)
    return (len(beta) == num_images) and isinstance(beta[0], Polynomial)


def test_standardize_beta():
    assert _beta_is_valid(2.5, 1)
    assert _beta_is_valid(2.5, 3)
    assert _beta_is_valid(Polynomial([0, 2.5]), 1)

    assert _beta_is_valid([2.0, 2.5], 2)
    assert _beta_is_valid(np.array([Polynomial([0, 2.5]), Polynomial([0, 2.5])]), 2)

    with pytest.raises(ValueError):
        # Wrong number of images requested
        _beta_is_valid([2.0, 2.5, 3.0, 3.0, 5], 2)

    # Check that we can pass iterables of coefficients
    assert _beta_is_valid([[0.0, 2.0], [0, 2.5]], 2)
    # Test cubic polys too
    assert _beta_is_valid([[0.0, 1.0, 2.0, 3.0]], 2)
    assert _beta_is_valid([[0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0]], 2)

    with pytest.raises(ValueError):
        # Wrong number of images requested
        assert _beta_is_valid([[0.0, 2.0], [0, 2.5]], 1)
        assert _beta_is_valid([[0.0, 2.0], [0, 2.5]], 3)


def test_beta_types():
    """ """
    # Scalar test
    b = 2.5
    beta_tol = 0.15  # For comparing two way simulate
    freq0 = 2.0e-3
    shape2d = (200, 200)
    out = turbulence.simulate(shape=shape2d, beta=b, freq0=freq0)
    assert out.shape == shape2d

    shape3d = (4, 200, 200)
    out = turbulence.simulate(shape=shape3d, beta=b, freq0=freq0)
    assert out.shape == shape3d
    psd = turbulence.Psd.from_image(out, freq0=freq0, deg=1)
    assert len(psd.beta) == shape3d[0]

    # Note that the slope will always be negative,
    # even though we allow user to pass positive sloeps
    bslope = -b
    for bh in psd.beta:
        assert abs(bh.coef[1] - bslope) < beta_tol

    # Same thing, but manually specifying same b
    out = turbulence.simulate(shape=shape3d, beta=[b, b, b, b], freq0=freq0)
    psd = turbulence.Psd.from_image(out, freq0=freq0, deg=1)
    for bh in psd.beta:
        assert abs(bh.coef[1] - bslope) < beta_tol

    beta_list = [-2.0, -2.3, -2.7, -2.9]
    out = turbulence.simulate(shape=shape3d, beta=beta_list, freq0=freq0)
    psd = turbulence.Psd.from_image(out, freq0=freq0, deg=1)
    for bh, b in zip(psd.beta, beta_list):
        assert abs(bh.coef[1] - b) < beta_tol


def test_load_save_psd(tmp_path):
    fname = tmp_path / "test_psd.npz"
    # Scalar test
    b = 2.5
    freq0 = 2.0e-3
    shape2d = (200, 200)
    out = turbulence.simulate(shape=shape2d, beta=b, freq0=freq0)
    psd = turbulence.Psd.from_image(out, freq0=freq0)
    psd.save(fname)
    psd2 = turbulence.Psd.load(fname)
    assert psd == psd2

    shape3d = (4, 200, 200)
    beta_list = [-2.0, -2.3, -2.7, -2.9]
    fname3d = tmp_path / "test_psd_3d.npz"

    out = turbulence.simulate(shape=shape3d, beta=beta_list, freq0=freq0)

    # p0, beta, f, psd = turbulence.get_psd(out, freq0=freq0, deg=3, filename=fname3d)
    # p0_2, beta_2, f_2, psd_2 = turbulence.get_psd(
    #     fname, freq0=freq0, deg=1, filename=fname3d
    # )


def test_max_amp():
    max_amp = 0.05  # meters
    noise = turbulence.simulate(shape=(100, 100), max_amp=max_amp)
    assert_allclose(np.max(noise), max_amp)
    assert noise.shape == (100, 100)

    noise3d = turbulence.simulate(shape=(3, 100, 100), max_amp=max_amp)
    assert noise3d.shape == (3, 100, 100)
    assert_allclose(np.max(noise3d, axis=(1, 2)), max_amp)


def test_resolution_freq_change():
    N = 200
    res = 60
    assert res == turbulence._resolution_from_freqs(turbulence._get_freqs(N, res))
