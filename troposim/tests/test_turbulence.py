from troposim import turbulence
from numpy.polynomial.polynomial import Polynomial
import numpy as np


class TestSimulate:
    """ """
    freq0 = 2.0e-3
    shape2d = (10, 10)
    shape3d = (4, 10, 10)

    def test_default(self):
        """ """
        turbulence.simulate()

    def test_shapes(self):
        """ """
        out = turbulence.simulate(shape=(10, 10), freq0=self.freq0)
        assert out.shape == (10, 10)
        out = turbulence.simulate(shape=(2, 10, 10), freq0=self.freq0)
        assert out.shape == (2, 10, 10)

    def test_beta_types(self):
        """ """
        # Scalar test
        b = 2.5
        out = turbulence.simulate(shape=self.shape2d, beta=b, freq0=self.freq0)
        assert out.shape == self.shape2d

        bpoly = Polynomial([1.0, 2.0, 3.3])
        out = turbulence.simulate(shape=self.shape2d, beta=bpoly, freq0=self.freq0)
        assert out.shape == self.shape2d

        beta = np.array([b, b, b, b])
        out = turbulence.simulate(shape=self.shape3d, beta=beta, freq0=self.freq0)
        assert out.shape == self.shape3d

        beta = [bpoly, bpoly, bpoly, bpoly]
        out = turbulence.simulate(shape=self.shape3d, beta=beta, freq0=self.freq0)
        assert out.shape == self.shape3d


def test_get_psd():
    """ """
    shape = (200, 200)
    p0 = 1e-3
    b = -2.5
    freq0 = 2.0e-3
    out = turbulence.simulate(shape=shape, beta=b, freq0=freq0, p0=p0)
    p0_hat, beta_hat, _, _ = turbulence.get_psd(out, freq0=freq0, deg=1)

    tol = 0.2
    assert abs(beta_hat.coef[1] - b) < tol
    assert abs(p0_hat - p0) < tol
