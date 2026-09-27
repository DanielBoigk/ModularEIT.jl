---
tags: [overview]
---

*Tomography* means reconstructing an image of an object's interior from many indirect measurements (projections) taken from outside. Each modality measures a different physical property:

| Modality | Physical quantity | Forward model |
|:--|:--|:--|
| X-ray computed tomography (CT) | X-ray attenuation coefficient | linear (Radon transform) |
| Magnetic resonance imaging (MRI) | proton density, relaxation times | approximately linear (Fourier sampling) |
| Ultrasound imaging | acoustic impedance, sound speed | wave equation |
| Diffuse optical / infrared tomography | optical absorption and scattering | diffusion equation, nonlinear |
| [[Electrical Impedance Tomography]] | electrical conductivity (admittivity) | elliptic PDE, nonlinear |

CT and MRI have forward models that are linear in the unknown and only mildly ill-posed, which is why they reach high resolution. EIT and diffuse optical tomography are *diffusive*: the signal spreads out through the whole body, the forward map is nonlinear, and the inverse problem is severely ill-posed (see [[Stability of the Calderón Problem]]). In exchange, EIT hardware is inexpensive, has high temporal resolution and uses no ionising radiation.

## References

1. M. Cheney, D. Isaacson, J. C. Newell (1999). *Electrical Impedance Tomography*. SIAM Review 41(1), 85–101. [doi:10.1137/S0036144598333613](https://doi.org/10.1137/S0036144598333613)
2. J. L. Mueller, S. Siltanen (2012). *Linear and Nonlinear Inverse Problems with Practical Applications*. SIAM. [doi:10.1137/1.9781611972344](https://doi.org/10.1137/1.9781611972344)
3. S. R. Arridge (1999). *Optical tomography in medical imaging*. Inverse Problems 15(2), R41–R93. [doi:10.1088/0266-5611/15/2/022](https://doi.org/10.1088/0266-5611/15/2/022)
