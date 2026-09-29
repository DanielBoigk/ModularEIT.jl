---
tags: [reconstruction, overview]
aliases: [Galerkin reconstruction, Reconstruction loop]
---

Most practical EIT reconstructions minimise a [[Variational Regularization|regularised objective]]

$$
\Phi(\sigma) = \sum_{i=1}^N J_i\big(u_i(\sigma)\big) + \beta\,\mathcal R(\sigma)
$$

iteratively, where $u_i(\sigma)$ solves the forward problem for [[Current Patterns|current pattern]] $g_i$. One iteration:

```text
given σ_k
1. assemble L_σk                                    (weighted stiffness matrix)
2. solve  L_σk u_i = g_i             for all i      (state equations)
3. misfit  J(σ_k) = Σ_i d(u_i|Γ, f_i)
4. solve  L_σk λ_i = ∂_u d(u_i|Γ, f_i)  for all i   (adjoint equations)
5. gradient  ∇J = Σ_i −∇u_i·∇λ_i                    (functional derivative)
6. add regulariser  β ∇R(σ_k)
7. update σ_{k+1} = σ_k + τ_k p_k                   (optimiser: GN, L-BFGS-B, ...)
until stopping criterion
```

The search direction $p_k$ is a descent direction, for example $p_k = -(\nabla J+\beta\nabla\mathcal R)$ for steepest descent. The step size $\tau_k>0$ comes from the optimiser (see [[Line Search]]).

Components and where they are described:

- steps 1–2: [[Weighted Stiffness Matrix]], [[Block Krylov Methods]];
- step 3: [[Data Fidelity Terms]];
- steps 4–5: [[Adjoint State Method]], [[Adjoint Equation]], [[Functional Derivative of the Data Misfit]];
- step 6: [[Tikhonov Regularization]], [[Smoothed Total Variation]], learned priors ([[Learned Regularization]]);
- step 7: [[Gauss-Newton Method]], [[Levenberg-Marquardt Method]], [[L-BFGS-B]], or splitting with [[ADMM]] (see [[Nested ADMM Reconstruction]]);
- stopping: [[Stopping Criteria]].

The per-pattern solves are independent, so the method parallelises naturally over patterns.

**In ModularEIT.jl:** [`minimize`](https://danielboigk.github.io/ModularEIT.jl/dev/api/optimization/#ModularEIT.minimize).

## References

1. M. Hinze, R. Pinnau, M. Ulbrich, S. Ulbrich (2009). *Optimization with PDE Constraints*. Springer. [doi:10.1007/978-1-4020-8839-1](https://doi.org/10.1007/978-1-4020-8839-1)
2. W. R. B. Lionheart (2004). *EIT reconstruction algorithms: pitfalls, challenges and recent developments*. Physiol. Meas. 25(1), 125–142. [doi:10.1088/0967-3334/25/1/021](https://doi.org/10.1088/0967-3334/25/1/021)
