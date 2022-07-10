import pytest
from numpy.testing import assert_approx_equal
from numpy.polynomial import Polynomial

from troposim import turbulence


class TestSimulate:
    beta_tol = 0.1  # For comparing two way simulate -> get_psd

    def test_default(self):
        turbulence.simulate()

    def test_shapes(self):
        freq0 = 2.0e-3
        out = turbulence.simulate(shape=(10, 10), freq0=freq0)
        assert out.shape == (10, 10)
        out = turbulence.simulate(shape=(2, 10, 10), freq0=freq0)
        assert out.shape == (2, 10, 10)

    def test_get_psd(self):
        """ """
        shape = (200, 200)
        p0 = 1e-3
        b = -2.5
        freq0 = 2.0e-3
        out = turbulence.simulate(shape=shape, beta=b, freq0=freq0, p0=p0)
        p0_hat, beta_hat, _, _ = turbulence.get_psd(out, freq0=freq0, deg=1)

        tol = 0.1
        assert abs(beta_hat.coef[1] - b) < tol
        assert abs(p0_hat - p0) < tol

    @staticmethod
    def _beta_is_valid(beta, num_images):
        beta = turbulence._standardize_beta(beta, num_images)
        return (len(beta) == num_images) and isinstance(beta[0], Polynomial)

    def test_standardize_beta(self):
        assert self._beta_is_valid(2.5, 1)
        assert self._beta_is_valid(2.5, 3)
        assert self._beta_is_valid([2.0, 2.5], 2)
        with pytest.raises(ValueError):
            # Wrong number of images requested
            self._beta_is_valid([2.0, 2.5, 3.0, 3.0, 5], 2)

        assert self._beta_is_valid([[0.0, 2.0], [0, 2.5]], 2)
        # Test cubic polys too
        assert self._beta_is_valid([[0.0, 1.0, 2.0, 3.0]], 2)
        assert self._beta_is_valid([[0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0]], 2)

        with pytest.raises(ValueError):
            # Wrong number of images requested
            assert self._beta_is_valid([[0.0, 2.0], [0, 2.5]], 1)
            assert self._beta_is_valid([[0.0, 2.0], [0, 2.5]], 3)

    def test_beta_types(self):
        """ """
        # Scalar test
        b = 2.5
        freq0 = 2.0e-3
        shape2d = (200, 200)
        out = turbulence.simulate(shape=shape2d, beta=b, freq0=freq0)
        assert out.shape == shape2d

        shape3d = (4, 200, 200)
        out = turbulence.simulate(shape=shape3d, beta=b, freq0=freq0)
        assert out.shape == shape3d
        _, beta_hat_list, _, _ = turbulence.get_psd(out, freq0=freq0, deg=1)
        assert len(beta_hat_list) == shape3d[0]

        # Note that the slope will always be negative,
        # even though we allow user to pass positive sloeps
        bslope = -b
        for bh in beta_hat_list:
            assert abs(bh.coef[1] - bslope) < self.beta_tol

        beta_list = [-2.0, -2.3, -2.7, -2.9]
        out = turbulence.simulate(shape=shape3d, beta=beta_list, freq0=freq0)
        _, beta_hat_list, _, _ = turbulence.get_psd(out, freq0=freq0, deg=1)
        for bh, b in zip(beta_hat_list, beta_list):
            assert abs(bh.coef[1] - b) < self.beta_tol

    def test_load_save_psd(self, tmp_path):
        fname = tmp_path / "test_psd.npz"
        # Scalar test
        b = 2.5
        freq0 = 2.0e-3
        shape2d = (200, 200)
        out = turbulence.simulate(shape=shape2d, beta=b, freq0=freq0)
        p0, beta, f, psd = turbulence.get_psd(out, freq0=freq0, deg=1, outname=fname)
        p0_2, beta_2, f_2, psd_2 = turbulence.load_psd(
            fname, freq0=freq0, deg=1, outname=fname
        )
        assert assert_approx_equal(p0, p0_2)
        assert assert_approx_equal(beta, beta_2)
        assert assert_approx_equal(f, f_2)
        assert assert_approx_equal(psd, psd_2)

        shape3d = (4, 200, 200)
        beta_list = [-2.0, -2.3, -2.7, -2.9]
        fname3d = tmp_path / "test_psd_3d.npz"

        out = turbulence.simulate(shape=shape3d, beta=beta_list, deg=3, freq0=freq0)

        p0, beta, f, psd = turbulence.get_psd(out, freq0=freq0, deg=3, outname=fname3d)
        p0_2, beta_2, f_2, psd_2 = turbulence.load_psd(
            fname, freq0=freq0, deg=1, outname=fname3d
        )
