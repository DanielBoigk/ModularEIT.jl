---
tags: [inverse-problem, reconstruction]
aliases: [Constructive reconstruction]
---

Some methods come with a mathematical guarantee that they recover $\gamma$ from $\Lambda_\gamma$, under assumptions. This is unlike generic optimisation, which only finds local minima.

- **Nachman (1988), $n\ge3$.** A constructive procedure based on [[Complex Geometrical Optics Solutions]]. A boundary integral equation gives the traces of the CGO solutions, these give the scattering transform of $q = \Delta\sqrt\gamma/\sqrt\gamma$, and from that $q$ and then $\gamma$ are recovered. Novikov (1988) obtained related results independently.
- **Nachman (1996), $n=2$.** A constructive proof for $\gamma\in W^{2,p}$ through a $\bar\partial$ ("D-bar") equation in the complex spectral parameter. This became the practical [[D-bar Method]]. Siltanen, Mueller and Isaacson (2000) gave the first numerical implementation. Knudsen, Lassas, Mueller and Siltanen (2009) proved that a truncated version is a regularisation strategy for noisy data.
- **Linearised and one-step methods.** Calderón's linearisation, NOSER, and series-reversion approaches expand the [[Forward Map]] around a reference conductivity. Garde, Hyvönen and Kuutela (2023) derived series reversion with local convergence guarantees, including modelling errors.
- **Monotonicity methods** recover the *shape* of inclusions with guarantees, using the order properties of $\Lambda_\gamma$ (see [[Properties of the Boundary Operators]]).

Iterative, regularised, optimisation-based methods are covered in [[Iterative Reconstruction Loop]]. They have no global guarantee but are flexible about electrodes, noise and prior information.

## References

1. A. I. Nachman (1988). *Reconstructions From Boundary Measurements*. Ann. of Math. 128(3), 531–576. [doi:10.2307/1971435](https://doi.org/10.2307/1971435)
2. R. G. Novikov (1988). *Multidimensional inverse spectral problem for the equation −Δψ + (v(x) − Eu(x))ψ = 0*. Funct. Anal. Appl. 22(4), 263–272. [doi:10.1007/BF01077418](https://doi.org/10.1007/BF01077418)
3. A. I. Nachman (1996). *Global Uniqueness for a Two-Dimensional Inverse Boundary Value Problem*. Ann. of Math. 143(1), 71–96. [doi:10.2307/2118653](https://doi.org/10.2307/2118653)
4. S. Siltanen, J. Mueller, D. Isaacson (2000). *An implementation of the reconstruction algorithm of A Nachman for the 2D inverse conductivity problem*. Inverse Problems 16(3), 681–699. [doi:10.1088/0266-5611/16/3/310](https://doi.org/10.1088/0266-5611/16/3/310)
5. K. Knudsen, M. Lassas, J. L. Mueller, S. Siltanen (2009). *Regularized D-bar method for the inverse conductivity problem*. Inverse Probl. Imaging 3(4), 599–624. [doi:10.3934/ipi.2009.3.599](https://doi.org/10.3934/ipi.2009.3.599)
6. H. Garde, N. Hyvönen, T. Kuutela (2023). *Series reversion for electrical impedance tomography with modeling errors*. Inverse Problems 39(8), 085007. [doi:10.1088/1361-6420/acdab8](https://doi.org/10.1088/1361-6420/acdab8)
7. B. Harrach, M. Ullrich (2013). *Monotonicity-based shape reconstruction in electrical impedance tomography*. SIAM J. Math. Anal. 45(6), 3382–3403. [doi:10.1137/120886984](https://doi.org/10.1137/120886984)
