---
tags: [inverse-problem]
aliases: [Inverse conductivity problem, Calderon problem, EIT inverse problem]
---

The **Calderón problem** (inverse conductivity problem) asks:

> Given the [[Dirichlet-to-Neumann Map]] $\Lambda_\gamma$, which is all boundary voltage–current pairs, determine the conductivity $\gamma$ in $\Omega$.

It was posed by A. P. Calderón in 1980. He also proved that the *linearised* problem around a constant conductivity is injective. The problem is the mathematical idealisation of [[Electrical Impedance Tomography]]: it assumes noise-free data and a [[Electrode Models|continuum of electrodes]].

The theory is organised around three questions, which mirror Hadamard's [[Well-Posedness|well-posedness]] conditions:

1. **Uniqueness**: does $\Lambda_{\gamma_1}=\Lambda_{\gamma_2}$ imply $\gamma_1=\gamma_2$? See [[Uniqueness in the Calderón Problem]] and [[Boundary Determination]].
2. **Reconstruction**: is there a constructive algorithm that maps $\Lambda_\gamma$ to $\gamma$? See [[Reconstruction Algorithms with Guarantees]] and the [[D-bar Method]].
3. **Stability**: how strongly do errors in $\Lambda_\gamma$ affect $\gamma$? See [[Stability of the Calderón Problem]].

The answers depend on the dimension, the regularity class of $\gamma$, and whether $\gamma$ is isotropic (see [[Anisotropic Conductivities]]). Central tools are [[Complex Geometrical Optics Solutions]] and the [[Linearized EIT and the Sensitivity Kernel|linearisation identity]].

In practice, noise and finitely many measurements make the inverse problem severely ill-posed. It is therefore solved as a regularised optimisation problem (see [[Variational Regularization]] and [[Iterative Reconstruction Loop]]).

## References

1. A. P. Calderón (1980/2006). *On an inverse boundary value problem*. Reprinted in Comput. Appl. Math. 25(2–3), 133–138. [doi:10.1590/S0101-82052006000200002](https://doi.org/10.1590/S0101-82052006000200002)
2. G. Uhlmann (2009). *Electrical impedance tomography and Calderón's problem*. Inverse Problems 25(12), 123011. [doi:10.1088/0266-5611/25/12/123011](https://doi.org/10.1088/0266-5611/25/12/123011)
3. J. L. Mueller, S. Siltanen (2012). *Linear and Nonlinear Inverse Problems with Practical Applications*. SIAM. [doi:10.1137/1.9781611972344](https://doi.org/10.1137/1.9781611972344)
