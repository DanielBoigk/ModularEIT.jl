---
tags: [forward-problem]
aliases: [Governing equation, Generalized Laplace equation]
---

Combining the steady-state [[Continuity Equation]] $\nabla\cdot\mathbf J = 0$ with [[Ohm's Law in Continuum Form|Ohm's law]] $\mathbf J = -\gamma\nabla u$ gives the **conductivity equation**

$$
\nabla\cdot(\gamma\nabla u) = 0 \qquad \text{in } \Omega ,
$$

for the electric potential $u:\Omega\to\mathbb R$ in a bounded Lipschitz domain $\Omega\subset\mathbb R^n$ ($n = 2,3$). The conductivity $\gamma\in L^\infty(\Omega)$ is assumed bounded above and below,

$$
0 < \gamma_{\min} \le \gamma(x) \le \gamma_{\max} < \infty \quad\text{a.e.},
$$

which makes the operator uniformly elliptic. For constant $\gamma$ the equation reduces to the Laplace equation $\Delta u = 0$.

On its own the equation has infinitely many solutions. A unique solution needs a boundary condition: either the voltage ([[Dirichlet Problem]]) or the current ([[Neumann Problem]]). Physically realistic electrodes lead to the Robin-type conditions of the [[Complete Electrode Model]].

The weak form is derived in [[Weak Formulation of the Conductivity Equation]]. Existence and uniqueness follow from the [[Lax-Milgram Theorem]].

## References

1. L. C. Evans (2010). *Partial Differential Equations*, 2nd ed. AMS Graduate Studies in Mathematics 19. [doi:10.1090/gsm/019](https://doi.org/10.1090/gsm/019)
2. L. Borcea (2002). *Electrical impedance tomography*. Inverse Problems 18(6), R99–R136. [doi:10.1088/0266-5611/18/6/201](https://doi.org/10.1088/0266-5611/18/6/201)
