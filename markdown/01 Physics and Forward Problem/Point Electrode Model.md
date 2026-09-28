---
tags: [forward-problem, electrodes]
aliases: [Point electrodes]
---

The **point electrode model** idealises every electrode as a single boundary point $x_\ell\in\partial\Omega$ (see [[Electrode Models]]). The current $I_\ell$ enters as a point source:

$$
\nabla\cdot(\sigma\nabla u) = 0 \ \text{ in }\Omega,\qquad \sigma\,\partial_\nu u = \sum_{\ell=1}^L I_\ell\,\delta_{x_\ell}\ \text{ on }\partial\Omega,\qquad \sum_\ell I_\ell = 0 .
$$

**Weak form.** For every test function $v$,

$$
\int_\Omega\sigma\nabla u\cdot\nabla v\,\mathrm dx = \sum_{\ell=1}^L I_\ell\,v(x_\ell).
$$

The right-hand side needs point values of $v$, which $H^1(\Omega)$ functions do not have in two or more dimensions. The point source is not in $H^{-1/2}(\partial\Omega)$, and the solution is not in $H^1(\Omega)$: near a source in a half-plane with constant $\sigma$, $u \approx -\frac{I_\ell}{\pi\sigma}\log|x-x_\ell|$.

**Consequence for measurements.** The potential is infinite at a current-carrying point electrode. Voltages can only be measured at points $x_m$ that carry no current. In a finite element discretisation the nodal value at an injection node stays finite, but it grows like $\log(1/h)$ under mesh refinement and has no physical meaning.

**Justification.** For small electrodes, the [[Complete Electrode Model]] and the point model give almost the same *relative* data (voltage differences between non-injecting electrodes). The error vanishes as the electrode size tends to zero. This makes point electrodes a useful model for small electrodes and a convenient setting for theory.

**Discretisation.** The load vector of electrode $\ell$ is the unit vector of the node closest to $x_\ell$, and measuring picks nodal values (see [[Discrete Electrode Models]]). The voltage-driven counterpart prescribes $u(x_\ell) = U_\ell$ at the electrode nodes and no current elsewhere. For consistent data it is the exact inverse of the current-driven problem.

## References

1. M. Hanke, B. Harrach, N. Hyvönen (2011). *Justification of point electrode models in electrical impedance tomography*. Math. Models Methods Appl. Sci. 21(6), 1395–1413. [doi:10.1142/S0218202511005362](https://doi.org/10.1142/S0218202511005362)
2. K.-S. Cheng, D. Isaacson, J. C. Newell, D. G. Gisser (1989). *Electrode models for electric current computed tomography*. IEEE Trans. Biomed. Eng. 36(9), 918–924. [doi:10.1109/10.35300](https://doi.org/10.1109/10.35300)
