---
tags: [forward-problem]
---

In the **Dirichlet problem** the voltage on the boundary is prescribed. Given $f\in H^{1/2}(\partial\Omega)$, find $u\in H^1(\Omega)$ with

$$
\nabla\cdot(\gamma\nabla u) = 0 \ \text{in } \Omega, \qquad u = f \ \text{on } \partial\Omega .
$$

**Weak form.** Find $u\in H^1(\Omega)$ with $\operatorname{tr}u = f$ such that

$$
\int_\Omega \gamma\,\nabla u\cdot\nabla v \,\mathrm dx = 0 \qquad \forall v\in H^1_0(\Omega).
$$

Writing $u = u_0 + E f$ with any extension $Ef\in H^1(\Omega)$ of $f$ turns this into a problem for $u_0\in H^1_0(\Omega)$. That problem has a unique solution by the [[Lax-Milgram Theorem]]; coercivity comes from the Poincaré inequality on $H^1_0$.

The measured quantity is the resulting boundary current $g = \gamma\,\partial_\nu u|_{\partial\Omega}\in H^{-1/2}(\partial\Omega)$. The map $f\mapsto g$ is the [[Dirichlet-to-Neumann Map]].

By the [[Dirichlet and Thomson Principles|Dirichlet principle]], $u$ minimises the power $\int_\Omega\gamma|\nabla v|^2$ among all $v$ with trace $f$.

In a finite element code, the boundary condition is imposed on the matrix as described in [[Enforcing Dirichlet Conditions]].

## References

1. L. C. Evans (2010). *Partial Differential Equations*, 2nd ed. AMS GSM 19. [doi:10.1090/gsm/019](https://doi.org/10.1090/gsm/019)
2. L. Borcea (2002). *Electrical impedance tomography*. Inverse Problems 18(6), R99–R136. [doi:10.1088/0266-5611/18/6/201](https://doi.org/10.1088/0266-5611/18/6/201)
