---
tags: [physics]
---

Conservation of electric charge is expressed by the **continuity equation**

$$
\frac{\partial \rho_q}{\partial t} + \nabla\cdot \mathbf J = 0 ,
$$

where $\rho_q$ is the volume charge density and $\mathbf J$ the current density. Integrated over a control volume, it says that the net current leaving the volume equals the rate at which the enclosed charge decreases. It is the continuum version of Kirchhoff's current law.

In [[Electrical Impedance Tomography]] no charge builds up inside the body and there are no interior sources, so in steady state

$$
\nabla\cdot\mathbf J = 0 \quad\text{in } \Omega .
$$

All current enters and leaves through the boundary. With [[Ohm's Law in Continuum Form|Ohm's law]] $\mathbf J = -\gamma\nabla u$, this becomes the [[Conductivity Equation]]. The fact that the total injected current must be zero (see [[Neumann Problem]]) follows from integrating $\nabla\cdot \mathbf J = 0$ over $\Omega$ with the divergence theorem.

## References

1. J. D. Jackson (1999). *Classical Electrodynamics*, 3rd ed., Wiley. [ISBN 978-0-471-30932-1](https://search.worldcat.org/search?q=bn:9780471309321)
2. M. Cheney, D. Isaacson, J. C. Newell (1999). *Electrical Impedance Tomography*. SIAM Review 41(1), 85–101. [doi:10.1137/S0036144598333613](https://doi.org/10.1137/S0036144598333613)
