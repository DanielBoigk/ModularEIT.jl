---
tags: [inverse-problem, forward-problem]
aliases: [Linearization, Fréchet derivative of the forward map, Sensitivity, Jacobian of EIT]
---

The [[Forward Map]] $\gamma\mapsto\Lambda_\gamma$ is Fréchet differentiable. Its derivative in direction $\delta\gamma$ is given by the **linearisation identity**

$$
\big\langle \Lambda'_\gamma[\delta\gamma]\, f,\ h\big\rangle = \int_\Omega \delta\gamma\ \nabla u_f\cdot\nabla u_h\,\mathrm dx ,
$$

where $u_f, u_h$ solve the [[Dirichlet Problem]] with data $f,h$. For the [[Neumann-to-Dirichlet Map]] the sign flips:

$$
\big\langle g,\ \mathcal R'_\gamma[\delta\gamma]\, k\big\rangle = -\int_\Omega \delta\gamma\ \nabla u_g\cdot\nabla u_k\,\mathrm dx .
$$

The product $\nabla u_g\cdot\nabla u_k$ is the **sensitivity kernel**. It tells how much the measurement "drive with $k$, read out with $g$" reacts to a local change of conductivity. It is large near the electrodes and decays towards the interior, which is another view of the ill-posedness.

**Derivation of the first identity.** Let $\dot u$ be the derivative of $u_f$ with respect to $\gamma$. Differentiating $\nabla\cdot(\gamma\nabla u_f)=0$ gives $\nabla\cdot(\gamma\nabla\dot u) = -\nabla\cdot(\delta\gamma\nabla u_f)$ with $\dot u|_{\partial\Omega}=0$. Testing with $u_h$ and using [[Green's Identities]] yields the formula.

**Uses.**

- Calderón proved injectivity of this linear map at $\gamma\equiv$ const, the first uniqueness result.
- One-step linear reconstruction methods (NOSER, GREIT; see [[EIT Software and Solvers]]) solve a regularised version of $\Lambda'_\gamma[\delta\gamma] \approx \Lambda_{\text{meas}}-\Lambda_{\gamma_0}$.
- The same product $\nabla u\cdot\nabla\lambda$ appears as the gradient in the [[Adjoint State Method]]. There $\lambda$ plays the role of the second field, driven by the data residual.
- Stacking the kernels for all pattern pairs gives the Jacobian used in the [[Gauss-Newton Method]].

## References

1. A. P. Calderón (1980/2006). *On an inverse boundary value problem*. Comput. Appl. Math. 25(2–3), 133–138. [doi:10.1590/S0101-82052006000200002](https://doi.org/10.1590/S0101-82052006000200002)
2. W. R. B. Lionheart (2004). *EIT reconstruction algorithms: pitfalls, challenges and recent developments*. Physiol. Meas. 25(1), 125–142. [doi:10.1088/0967-3334/25/1/021](https://doi.org/10.1088/0967-3334/25/1/021)
3. L. Borcea (2002). *Electrical impedance tomography*. Inverse Problems 18(6), R99–R136. [doi:10.1088/0266-5611/18/6/201](https://doi.org/10.1088/0266-5611/18/6/201)
