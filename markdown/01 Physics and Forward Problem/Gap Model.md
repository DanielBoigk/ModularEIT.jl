---
tags: [forward-problem, electrodes]
aliases: [Gap electrode model]
---

The **gap model** spreads the current of each electrode $e_\ell\subset\partial\Omega$ uniformly over the electrode and assumes no current in the gaps between electrodes (see [[Electrode Models]]):

$$
\nabla\cdot(\sigma\nabla u) = 0\ \text{ in }\Omega,\qquad
\sigma\,\partial_\nu u = \begin{cases} I_\ell/|e_\ell| & \text{on } e_\ell,\\ 0 & \text{on }\partial\Omega\setminus\bigcup_\ell e_\ell,\end{cases}\qquad \sum_\ell I_\ell = 0 .
$$

It is a [[Neumann Problem]] with piecewise constant data, so it is well posed up to an additive constant (see [[Grounding of the Potential]]). The voltage of electrode $m$ is modelled as the **mean potential** on it:

$$
V_m = \frac1{|e_m|}\int_{e_m} u\,\mathrm ds .
$$

**Reciprocity.** If the same electrodes inject and measure, the map $I\mapsto V$ is symmetric: the current weights $1/|e_\ell|$ on $e_\ell$ and the averaging weights coincide. It is positive semidefinite, and $I^\top V = \int_\Omega\sigma|\nabla u|^2$ is the dissipated power.

**Limitations.** A metal electrode is a good conductor, so its surface is (nearly) an equipotential and the current density is *not* uniform. It concentrates at the electrode edges (see [[Shunt Model]]). The gap model ignores this and also ignores the contact impedance. It systematically overestimates the resistivity, and the voltages it predicts on current-carrying electrodes miss the contact voltage drop entirely. Voltages on electrodes without current are much less affected (see [[Measurement Protocols]]).

**Injection and measurement electrodes may differ.** Current can be driven through one set of electrodes while voltages are averaged over another set. Then the input-output map is no longer square.

**Discretisation.** With the finite element basis $\varphi_i$, the load vector of electrode $\ell$ is $\int_{e_\ell}\varphi_i\,\mathrm ds/|e_\ell|$, and the measurement matrix is its transpose (see [[Discrete Electrode Models]]).

## References

1. K.-S. Cheng, D. Isaacson, J. C. Newell, D. G. Gisser (1989). *Electrode models for electric current computed tomography*. IEEE Trans. Biomed. Eng. 36(9), 918–924. [doi:10.1109/10.35300](https://doi.org/10.1109/10.35300)
2. E. Somersalo, M. Cheney, D. Isaacson (1992). *Existence and Uniqueness for Electrode Models for Electric Current Computed Tomography*. SIAM J. Appl. Math. 52(4), 1023–1040. [doi:10.1137/0152060](https://doi.org/10.1137/0152060)
