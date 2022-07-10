import numpy as np
from numpy.polynomial.polynomial import Polynomial

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

    def test_get_psd():
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

    def test_beta_types(self):
        """ """
        # Scalar test
        b = 2.5
        freq0 = 2.0e-3
        shape2d = (200, 200)
        out = turbulence.simulate(shape=shape2d, beta=b, freq0=freq0)
        assert out.shape == shape2d

        bpoly = Polynomial([1.0, 2.0, 3.3])
        out = turbulence.simulate(shape=shape2d, beta=bpoly, freq0=freq0)
        assert out.shape == shape2d

        shape3d = (4, 200, 200)
        beta = np.array([b, b, b, b])
        out = turbulence.simulate(shape=shape3d, beta=beta, freq0=freq0)
        assert out.shape == shape3d
        _, beta_hat, _, _ = turbulence.get_psd(out, freq0=freq0, deg=1)
        assert abs(beta_hat.coef[1] - b) < self.beta_tol

        beta = [bpoly, bpoly, bpoly, bpoly]
        out = turbulence.simulate(shape=shape3d, beta=beta, freq0=freq0)
        assert out.shape == shape3d

        beta_list = [2.0, 2.3, 2.7, 2.9]
        out = turbulence.simulate(shape=shape3d, beta=beta_list, freq0=freq0)
        _, beta_hat_list, _, _ = turbulence.get_psd(out, freq0=freq0, deg=1)
        for bh, b in zip(beta_hat_list, beta_list):
            assert abs(bh.coef[1] - b) < self.beta_tol