---
tags: [outlook, software]
---

Lessons for building a fast, modular EIT reconstruction library. They follow from the structure of the problem described in this wiki.

**Exploit the pattern-level parallelism.** All state and adjoint solves of one iteration share a single matrix $L_\sigma$ (see [[Adjoint State Method]]).

- Batch them as block solves (block CG or block MINRES; see [[Block Krylov Methods]]) and run them on the GPU.
- Solve state and adjoint systems in the same batch once residuals are available.
- Assemble the gradient contributions $-\nabla u_i\cdot\nabla\lambda_i$ in parallel as well.

**Split the work between CPU and GPU.** Sparse assembly of $L_\sigma$, setup of the [[Algebraic Multigrid|preconditioner]], mesh adaptation, and processing of the incoming residuals and gradients suit the CPU. Bulk linear algebra suits the accelerator. Avoid global synchronisation points: a Gauss–Newton solve that waits for all patterns stalls the pipeline, so asynchronous or stochastic updates over patterns are worth considering.

**Control accuracy and error.** Tie linear solver tolerances to the optimisation progress. Estimate discretisation errors (see [[A Posteriori Error Estimation and Adaptive Meshing]]). Adapt meshes near electrodes and conductivity jumps. Always [[Gradient Testing|test gradients]].

**Keep components exchangeable.** Forward discretisation (FEM, spectral), metric ([[Data Fidelity Terms]]), regulariser (classical or learned), and optimiser ([[L-BFGS-B]], [[Gauss-Newton Method|Gauss–Newton]], [[ADMM]]) should be independent modules behind small interfaces.

**Prefer schedule-free learned priors.** Priors that do not need a synchronised diffusion time ([[RED-Diff]], [[Diffusion Proximal Operator]]) can run concurrently with the physics solver. Guidance-based samplers interleave the two tightly.

**Respect the geometry.** Generic regularisers such as TV and Tikhonov ignore the structure of EIT. Spectral information from the boundary operator ([[Truncated SVD Regularization]]) and the [[Symmetries of the EIT Problem]] are underused sources of prior knowledge.

## References

1. A. Adler, W. R. B. Lionheart (2006). *Uses and abuses of EIDORS: an extensible software base for EIT*. Physiol. Meas. 27(5), S25–S42. [doi:10.1088/0967-3334/27/5/S03](https://doi.org/10.1088/0967-3334/27/5/S03)
2. W. R. B. Lionheart (2004). *EIT reconstruction algorithms: pitfalls, challenges and recent developments*. Physiol. Meas. 25(1), 125–142. [doi:10.1088/0967-3334/25/1/021](https://doi.org/10.1088/0967-3334/25/1/021)
