---
tags: [physics]
aliases: [Admittivity]
---

Biological tissue both conducts and stores charge: cell membranes act as capacitors. Under a time-harmonic excitation with angular frequency $\omega$ the material is described by the complex **admittivity**

$$
\gamma(x,\omega) = \sigma(x) + i\,\omega\,\varepsilon(x),
$$

where $\sigma$ is the conductivity and $\varepsilon$ the permittivity. The potential then solves $\nabla\cdot(\gamma\nabla u)=0$ with complex $\gamma$ (see [[Quasi-Static Approximation]]).

Consequences:

- Voltages and currents have an amplitude *and* a phase. The system behaves like a linear time-invariant (LTI) system at each frequency, while $\gamma\mapsto$ data stays nonlinear.
- *Multi-frequency* and *frequency-difference* EIT use the fact that $\sigma$ and $\varepsilon$ of tissues depend on frequency in characteristic ways (tissue spectroscopy). This makes reconstructions possible without a reference measurement.
- For $\omega\varepsilon\ll\sigma$ the imaginary part is negligible and one recovers the real [[Conductivity Equation]].

## References

1. M. Cheney, D. Isaacson, J. C. Newell (1999). *Electrical Impedance Tomography*. SIAM Review 41(1), 85–101. [doi:10.1137/S0036144598333613](https://doi.org/10.1137/S0036144598333613)
2. A. Adler, A. Boyle (2017). *Electrical Impedance Tomography: Tissue Properties to Image Measures*. IEEE Trans. Biomed. Eng. 64(11), 2494–2504. [doi:10.1109/TBME.2017.2728323](https://doi.org/10.1109/TBME.2017.2728323)
