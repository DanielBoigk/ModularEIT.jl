---
tags: [forward-problem, electrodes]
aliases: [Shunt electrode model]
---

The **shunt model** treats each electrode $e_\ell$ as a perfect conductor. The potential is constant on the electrode, and only the total current through it is prescribed (see [[Electrode Models]]):

$$
\begin{aligned}
\nabla\cdot(\sigma\nabla u) &= 0 && \text{in }\Omega,\\
u &= U_\ell && \text{on } e_\ell,\\
\int_{e_\ell}\sigma\,\partial_\nu u\,\mathrm ds &= I_\ell && \ell = 1,\dots,L,\\
\sigma\,\partial_\nu u &= 0 && \text{on }\partial\Omega\setminus\textstyle\bigcup_\ell e_\ell .
\end{aligned}
$$

The electrode voltages $U_\ell$ are unknowns, as in the [[Complete Electrode Model]]. The shunt model is the limit $z_\ell\to 0$ of the CEM (no contact impedance).

**Edge singularity.** At the boundary of an electrode, the boundary condition switches from Dirichlet to Neumann. There the current density is singular, like $r^{-1/2}$ in the distance $r$ to the electrode edge. Most of the current enters near the electrode edges, not uniformly as the [[Gap Model]] assumes. A positive contact impedance removes this singularity, which is one reason to prefer the CEM.

**Voltage-driven form.** Prescribing the electrode voltages $U$ instead of the currents gives a mixed boundary value problem: Dirichlet on the electrodes, homogeneous Neumann in the gaps. It is uniquely solvable without grounding, and the currents follow as $I_\ell = \int_{e_\ell}\sigma\,\partial_\nu u$. Discretely, $u$ is fixed on all nodes of $e_\ell$, and $I_\ell$ is the sum of the residual of the discrete equation over these nodes (see [[Discrete Electrode Models]]).

**The gap and shunt models are different models.** The voltage-driven shunt problem is *not* the inverse of the current-driven gap model: the gap model has a uniform current density with a non-constant potential on the electrode, the shunt model the reverse. Functionals that compare a current-driven and a voltage-driven solution of the same data, such as the [[Kohn-Vogelius Functional]], therefore do not vanish at the true conductivity if these two models are paired.

## References

1. K.-S. Cheng, D. Isaacson, J. C. Newell, D. G. Gisser (1989). *Electrode models for electric current computed tomography*. IEEE Trans. Biomed. Eng. 36(9), 918–924. [doi:10.1109/10.35300](https://doi.org/10.1109/10.35300)
2. E. Somersalo, M. Cheney, D. Isaacson (1992). *Existence and Uniqueness for Electrode Models for Electric Current Computed Tomography*. SIAM J. Appl. Math. 52(4), 1023–1040. [doi:10.1137/0152060](https://doi.org/10.1137/0152060)
