---
tags: [state-of-the-art, software]
---

A short survey of established numerical EIT solvers.

**Classical algorithms**

- **NOSER** (Cheney et al. 1990): a one-step Newton method with a diagonal regulariser. It linearises around the best-fitting constant conductivity.
- **GREIT** (Adler et al. 2009): a consensus linear reconstruction algorithm for 2D lung EIT. The reconstruction matrix is trained on simulated targets to meet figures of merit such as uniform amplitude response, low position error and small blur.
- **Iterative Gauss–Newton** with Tikhonov or [[Total Variation]] regularisation (see [[Gauss-Newton Method]]).
- **[[D-bar Method]]**: direct nonlinear 2D reconstruction.
- **Statistical inversion** with MCMC (Kaipio et al. 2000; see [[Bayesian Inversion]]).

**Software packages**

- **EIDORS**: the reference MATLAB/Octave toolbox for forward modelling and reconstruction with the [[Complete Electrode Model]] (Adler & Lionheart 2006).
- **pyEIT**: a Python framework with mesh generation, forward solvers and common reconstruction algorithms (Liu et al. 2018).
- **eit_fenicsx**: FEniCSx-based EIT forward and inverse solvers by A. Denker.
- **JEL.jl**: a Julia EIT library (Dizon, Jauhiainen, Valkonen).
- **ModularEIT.jl**: a Julia library on Ferrite.jl with adjoint gradients and pluggable regularisers and optimisers (this project).

For learning-based methods see [[Deep Learning for EIT]] and [[Diffusion Models for EIT]].

## References

1. M. Cheney, D. Isaacson, J. C. Newell, S. Simske, J. Goble (1990). *NOSER: An algorithm for solving the inverse conductivity problem*. Int. J. Imaging Syst. Technol. 2(2), 66–75. [doi:10.1002/ima.1850020203](https://doi.org/10.1002/ima.1850020203)
2. A. Adler et al. (2009). *GREIT: a unified approach to 2D linear EIT reconstruction of lung images*. Physiol. Meas. 30(6), S35–S55. [doi:10.1088/0967-3334/30/6/S03](https://doi.org/10.1088/0967-3334/30/6/S03)
3. A. Adler, W. R. B. Lionheart (2006). *Uses and abuses of EIDORS: an extensible software base for EIT*. Physiol. Meas. 27(5), S25–S42. [doi:10.1088/0967-3334/27/5/S03](https://doi.org/10.1088/0967-3334/27/5/S03)
4. B. Liu et al. (2018). *pyEIT: A python based framework for Electrical Impedance Tomography*. SoftwareX 7, 304–308. [doi:10.1016/j.softx.2018.09.005](https://doi.org/10.1016/j.softx.2018.09.005)
5. J. P. Kaipio, V. Kolehmainen, E. Somersalo, M. Vauhkonen (2000). *Statistical inversion and Monte Carlo sampling methods in electrical impedance tomography*. Inverse Problems 16(5), 1487–1522. [doi:10.1088/0266-5611/16/5/321](https://doi.org/10.1088/0266-5611/16/5/321)
6. A. Denker. *eit_fenicsx* (software). [github.com/alexdenker/eit_fenicsx](https://github.com/alexdenker/eit_fenicsx)
7. N. Dizon, J. Jauhiainen, T. Valkonen. *JEL.jl* (software). [zenodo.org/records/15028468](https://zenodo.org/records/15028468)
8. D. Boigk. *ModularEIT.jl* (software). [github.com/DanielBoigk/ModularEIT.jl](https://github.com/DanielBoigk/ModularEIT.jl)
