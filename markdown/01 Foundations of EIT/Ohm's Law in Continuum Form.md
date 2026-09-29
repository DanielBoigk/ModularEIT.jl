---
tags: [physics]
aliases: [Ohm's law]
---

For a resistor, Ohm's law $U = R\,I$ relates the voltage $U$ to the current $I$ through the resistance $R$. Inside a continuous medium it takes a local form. It links the **current density** $\mathbf J(x)$ (current per unit area perpendicular to the flow, in A/m²) to the **electric field** $\mathbf E(x)$:

$$
\mathbf J = \gamma\,\mathbf E ,
$$

where $\gamma(x) > 0$ is the **conductivity** (S/m) and $\rho = 1/\gamma$ is the resistivity (Ω·m).

Under the [[Quasi-Static Approximation]] the electric field is curl-free, so it is the negative gradient of a scalar electric potential $u$ (the voltage):

$$
\mathbf E = -\nabla u, \qquad \mathbf J = -\gamma \nabla u .
$$

The minus sign says that current flows from high to low potential. For anisotropic media such as muscle fibres, $\gamma$ is a symmetric positive definite matrix field instead of a scalar (see [[Anisotropic Conductivities]]).

Combined with the [[Continuity Equation]], this gives the [[Conductivity Equation]].

## References

1. J. D. Jackson (1999). *Classical Electrodynamics*, 3rd ed., Wiley. [ISBN 978-0-471-30932-1](https://search.worldcat.org/search?q=bn:9780471309321)
2. M. Cheney, D. Isaacson, J. C. Newell (1999). *Electrical Impedance Tomography*. SIAM Review 41(1), 85–101. [doi:10.1137/S0036144598333613](https://doi.org/10.1137/S0036144598333613)
