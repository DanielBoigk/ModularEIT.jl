---
tags: [physics]
---

EIT systems inject alternating currents at frequencies from roughly 1 kHz to 1 MHz. At these frequencies the wavelength of electromagnetic waves is far larger than the body, and magnetic induction is negligible. Maxwell's equations then reduce to their **quasi-static** form:

- $\nabla\times\mathbf E \approx 0$, so $\mathbf E = -\nabla u$ for a scalar potential $u$;
- the displacement and conduction currents together are divergence-free.

With time-harmonic excitation $e^{i\omega t}$, the conductivity is replaced by the complex **admittivity** $\gamma = \sigma + i\omega\varepsilon$ (see [[Complex Conductivity]]). The potential then solves $\nabla\cdot(\gamma\nabla u) = 0$ with complex coefficients.

In the **DC (or low-frequency) limit** $\omega\varepsilon \ll \sigma$, the admittivity is real and the model reduces to the real [[Conductivity Equation]]. This is the setting of most of this wiki.

## References

1. M. Cheney, D. Isaacson, J. C. Newell (1999). *Electrical Impedance Tomography*. SIAM Review 41(1), 85–101. [doi:10.1137/S0036144598333613](https://doi.org/10.1137/S0036144598333613)
2. E. Somersalo, M. Cheney, D. Isaacson (1992). *Existence and Uniqueness for Electrode Models for Electric Current Computed Tomography*. SIAM J. Appl. Math. 52(4), 1023–1040. [doi:10.1137/0152060](https://doi.org/10.1137/0152060)
