# Disambiguating terms from fractal surfaces, Gaussian random fields, and power laws

**Driving question**: 

Is the fractal atmospheric surface (a type of spatially correlated noise), which has a power-law power spectrum of

$$S(f) = (1/f)^{\beta}$$
also a Gaussian random field? 

The three related subjects are... 
- Gaussian Fields
- Fractal surfaces
- Power-law power spectra
Can all three be present? Or are some mutually exclusive?


**POSSIBLE SUMMARY ANSWER**

Yes, the surface can be
1. fractal, since the structure function (aka semi-variogram) of the atmospheric delay has a power-law form (Hanssen, Eq 4.7.10) [^1] :

$$D_s(\rho) \sim \rho^{5/3}$$

2. A Gaussian field, since by (Cressie, 1993) Eq. 5.5.1 [^2],  

More generally, a fractional Brownian motion in $R^d$ is a Gaussian process $Z(\cdot)$ characterized by a covariance function of the form with


$$C(s, t) = \frac{1}{2}\left(s^{2H} + t^{2H} + |s-t|^{2H}\right)$$

and a variogram of the form

$$E[(Z(s + h) - Z(s))^2] \propto ||h||^{2H}$$
So, for the Hanssen structure fucntion, $H=5/6$

3. A signal power law power spectrum $S(k)$ of the form (Hanssen Eq. 4.7.12)

$$P(k) = P_0 \left(f/f_0\right)^{-\beta}$$
where $P_0$ is the power at reference frequency $f_0$ and $\beta$ is called the *spectral index*. 
The structure function of a signal with this power spectrum can be written, using the same $\beta$, as
$$D_{\phi}(\rho) \propto \rho^{\beta-1}$$
which means that if the structure function exponent is $5/3$, then $\beta=8/3$ for the spectrum slope.

My confusion earlier: thinking that the frequency content of the signal was doing to dictate the amplitude distribution in space (or time for 1D)... In the same way that noise can be white (a frequency/spectrum description) and either uniform or Gaussian (a time or space description), the random process can be a Gaussian process, but have the same power-law exponent.

LAST QUESTIONs:
- BIGGEST UNKNOWN: Hanssen says the structure functions/SMV is $\rho^{5/3}$, which means in never levels off, and he says the covariance function doesn't exist...
    - Does this imply that it can't be a Gaussian field?
    - **ANSWER**: most places seem to say that fractional brownian motion is still a Gaussian field (Brownian sheet having covariance = min(s, t))
    - **TODO** This means it has stationary increments, but isn't 2nd order stationary...  I thought other places said "Gaussian random fields are 2nd order stationary (and sometimes strictly stationary under some condition)"
    - or does out limit study area + ramp removal mean in practice it levels off...
- if a random field is Gaussian, does it's structure func/SMV have to be gaussian?.... NO. thats related to the Covariance FUNCTION... aka, the covariance of the Gaussian variables at a given distance apart! the further away, the different the covariance is... but the entire process is still only characterized by it's mean (assumed 0), and covariance function (related to the structure function)
    - SO, a power-law SMV is fine for a Gaussian process. and it will have a power law power-spectrum by FT: https://math.stackexchange.com/questions/2173780/computing-fourier-transform-of-power-law

- I thought that Gaussian processes had have Gaussian Fourier transforms... 
    - answer? the power spectrum relates to the FT of the covariance function of a (second order stationary process) (or, with different terms, ACF <-> PSD)
- so i think "fractal" descriptions are orthogonal to both "Gaussian process" and "Power spectrum" shape... they might just be 3 completely separate axes, where none imply others

# Definitions
From Hanssen, 4.7 intro:
> The behavior of atmospheric signal in radar interferograms can be mathematically described using several interrelated measures such as the power spectrum, the covariance function, the structure function, and the fractal dimension.

> The power spectrum is useful to recognize scaling properties of the data or to distinguish different scaling regimes.

TODO
1. Random field
1. Gaussian random field
1. Power spectrum

[^1]: Hanssen, Ramon F. Radar interferometry: data interpretation and error analysis. Vol. 2. Springer Science & Business Media, 2001.

[^2]: Cressie, Noel. Statistics for spatial data. John Wiley & Sons, 2015.
