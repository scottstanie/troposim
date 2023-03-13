import numpy as np
import pytest
from numpy.polynomial import Polynomial
from numpy.testing import assert_allclose

from troposim import turbulence


def test_default():
    turbulence.simulate()


def test_shapes():
    freq0 = 2.0e-3
    out = turbulence.simulate(shape=(10, 10), freq0=freq0)
    assert out.shape == (10, 10)
    out = turbulence.simulate(shape=(2, 10, 10), freq0=freq0)
    assert out.shape == (2, 10, 10)
    out_rect = turbulence.simulate(shape=(3, 10, 20), freq0=freq0)
    assert out_rect.shape == (3, 10, 20)


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


def test_zero_psd():
    """ """
    shape = (200, 200)
    turbulence.Psd.from_image(np.zeros(shape), deg=1)
    turbulence.Psd.from_image(np.zeros(shape), deg=3)


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


def test_from_hdf5(tmp_path):
    import h5py

    noise4 = turbulence.simulate(shape=(4, 200, 200))
    fname = tmp_path / "test_psd.h5"
    dset = "noise"
    with h5py.File(fname, "w") as hf:
        hf.create_dataset(dset, data=noise4)
    psd4 = turbulence.Psd.from_hdf5(fname, dset, resolution=60)
    assert psd4.p0.shape == (4,)
    assert psd4.beta.shape == (4,)
    assert psd4.freq.shape == (99,)
    assert psd4.psd1d.shape == (4, 99)


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


def test_get_auto_freq0():
    res = 100
    noise = turbulence.simulate(shape=(200, 200), resolution=res)
    psd = turbulence.Psd.from_image(noise, resolution=res)

    assert psd._get_freq0(psd.freq, None) == 1e-4
    assert psd._get_freq0(psd.freq, 1e-4) == 1e-4
    assert psd._get_freq0(psd.freq, 1e-3) == 1e-3
    with pytest.raises(ValueError):
        assert psd._get_freq0(psd.freq, 1e-6)  # too small
        assert psd._get_freq0(psd.freq, 1)  # too big

    # Check the auto-picking for a small image
    noise = turbulence.simulate(shape=(50, 50), resolution=res, p0=1, freq0=1e-3)
    psd = turbulence.Psd.from_image(noise, resolution=res)
    assert psd._get_freq0(psd.freq, None) == 2e-4
    psd = turbulence.Psd.from_image(noise[:25, :25], resolution=res)
    assert psd._get_freq0(psd.freq, None) == 4e-4


def test_append_psds():
    noise = turbulence.simulate(shape=(200, 200))
    noise2 = turbulence.simulate(shape=(200, 200))

    psd = turbulence.Psd.from_image(noise, resolution=60)
    psd2 = turbulence.Psd.from_image(noise2, resolution=60)
    assert len(psd) == len(psd2) == 1
    psd.append(psd2)
    assert len(psd) == 2
    psd.append(psd)
    assert len(psd) == 4

    assert psd.p0.shape == (4,)
    assert psd.beta.shape == (4,)
    assert psd.freq.shape == (99,)
    assert psd.psd1d.shape == (4, 99)

    psd_wrongshape = turbulence.Psd.from_image(turbulence.simulate(shape=(250, 250)))
    with pytest.raises(ValueError):
        psd.append(psd_wrongshape)


@pytest.fixture
def psd5():
    return turbulence.Psd.from_image(turbulence.simulate(shape=(5, 200, 200)))


def test_len_psds(psd5):
    assert len(psd5) == 5


def test_getitem(psd5):
    n = 1
    assert len(psd5[0]) == n
    assert psd5[0].p0.shape == (n,)
    assert psd5[0].beta.shape == (n,)
    assert psd5[0].freq.shape == (99,)
    assert psd5[0].psd1d.shape == (n, 99)

    n = 3
    psd_slice = psd5[:n]
    assert len(psd_slice) == n
    assert psd_slice.p0.shape == (n,)
    assert psd_slice.beta.shape == (n,)
    assert psd_slice.freq.shape == (99,)
    assert psd_slice.psd1d.shape == (n, 99)


def test_dict_roundtrip(psd5):
    psd_dict = psd5.asdict()
    psd5b = turbulence.Psd.from_dict(psd_dict)
    assert psd5 == psd5b


# TODO: something weird with a 0 one
def test_zero_img():
    out = turbulence.simulate(
        shape=(3, 100, 100), beta=[2.0, 2.7, 3.0], verbose=True, freq0=1e-3
    )
    out[1] = 0
    psd = turbulence.Psd.from_image(out)
    psd.simulate()
