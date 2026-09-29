---
tags: [forward-problem]
aliases: [Forward problem, Forward operator]
---

The **forward map** of EIT takes a conductivity to the boundary operator it generates:

$$
\mathcal F:\ \gamma \mapsto \Lambda_\gamma \quad(\text{or } \gamma\mapsto\mathcal R_\gamma = \Lambda_\gamma^{-1}).
$$

For a finite set of current patterns $g_1,\dots,g_N$ it becomes the finite-dimensional map $\gamma\mapsto (\mathcal R_\gamma g_i)_{i=1}^N$ into voltages.

**Nonlinearity.** $\mathcal F$ is positively homogeneous of degree one, $\Lambda_{c\gamma} = c\Lambda_\gamma$, but not additive. In general $\Lambda_{\gamma_1+\gamma_2}\neq\Lambda_{\gamma_1}+\Lambda_{\gamma_2}$, because the potential $u$ itself depends on $\gamma$. So $\mathcal F$ is nonlinear. It is, however, smooth (Fréchet differentiable, even analytic) on the set of conductivities bounded away from zero. Its derivative is described in [[Linearized EIT and the Sensitivity Kernel]].

**Smoothing.** Small, fine-scale or deep changes in $\gamma$ produce very small changes in the boundary data. This is the source of the severe ill-posedness of the inverse direction (see [[Stability of the Calderón Problem]]).

**Evaluation.** Numerically, $\mathcal F$ is evaluated by solving one [[Neumann Problem]] (or [[Dirichlet Problem]]) per current pattern, usually with the [[Galerkin Method|finite element method]]. All patterns share the same system matrix $L_\gamma$ (see [[Weighted Stiffness Matrix]]), so they can be solved together as a block system (see [[Block Krylov Methods]]).

Inverting $\mathcal F$ is the [[Calderón Problem]].

**In ModularEIT.jl:** [`ForwardModel`](https://danielboigk.github.io/ModularEIT.jl/dev/api/forward/#ModularEIT.ForwardModel), [`forward_neumann`](https://danielboigk.github.io/ModularEIT.jl/dev/api/forward/#ModularEIT.forward_neumann), [`forward_dirichlet`](https://danielboigk.github.io/ModularEIT.jl/dev/api/forward/#ModularEIT.forward_dirichlet).

## References

1. L. Borcea (2002). *Electrical impedance tomography*. Inverse Problems 18(6), R99–R136. [doi:10.1088/0266-5611/18/6/201](https://doi.org/10.1088/0266-5611/18/6/201)
2. J. L. Mueller, S. Siltanen (2012). *Linear and Nonlinear Inverse Problems with Practical Applications*. SIAM. [doi:10.1137/1.9781611972344](https://doi.org/10.1137/1.9781611972344)
