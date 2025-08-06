# troposim

Simulate tropospheric noise for InSAR data

![](docs/example.jpg)

## Installation

```bash
pip install troposim
```

To make an editable installation

```bash
git clone https://github.com/scottstanie/troposim && cd troposim
pip install -e .
```

## Usage

To simulate one turbulence image, you can specify the shape:
```python
from troposim import turbulence
noise = turbulence.simulate(shape=(500, 500))
```
or add a 3rd dimension to simulate a stack of images

```python
noise = turbulence.simulate(shape=(10, 500, 500))
```

The `beta` argument is the slope of the log10(power) vs log10(frequency) graph.
The default is to use a single linear slope of $\beta = 8 / 3$:

$$
P(f) \propto \frac{1}{f^\beta}
$$

For smaller-scale turbulence, you can use a different `beta`:
```python
flatter_noise = turbulence.simulate(beta=2.2)
```

Since real InSAR data typically have a power spectrum that is not a single slope, you can **estimate the spectrum from an image** and use that to simulate new data:
```python
from troposim.turbulence import Psd
psd = Psd.from_image(noise)
new_noise = psd.simulate()
```
Here the `psd` object has attributes
- `p0`: the power at the reference frequency `freq0`
- `beta`: a numpy Polynomial which was fit to the log-log PSD
along with `psd1d`, which are the radially averaged spectrum values at the `psd.freq` frequencies. You can see these with the `.plot()` method.

```python
# assuming maptlotlib is installed
psd.plot()

# Or, to plot a side-by-side of image and 1D PSD
from troposim import plotting 
plotting.plot_psd(noise, freq=freq, psd1d=psd1d)
# Or just the PSD plot, no image
plotting.plot_psd1d(psd.freq, psd.psd1d)
```

To simulate a stack of new values from the PSD of one image, you simply pass in a new `shape` argument to `.simulate`:
```python
psd.simulate(shape=(10, 400, 400))
```
Note that the default fit will use a cubic polynomial. 
To request only a linear fit,
```python
psd = Psd.from_image(noise, deg=1)
```

You can also save the PSD parameters for later use:
```python
psd.save(outfile="my_psd.npz")
# Later, reload from this file
psd = Psd.load(outfile)
```


## Citation

If you find this library useful, [please consider citing our paper](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2024JB029614):

> Staniewicz, S., & Chen, J. (2025). Automatic detection of InSAR deformation and tropospheric noise features using computer vision: A case study over West Texas. Journal of Geophysical Research: Solid Earth, 130(7), e2024JB029614. 
