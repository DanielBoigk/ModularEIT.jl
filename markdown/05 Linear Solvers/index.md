---
title: Linear Solvers
tags: [overview, linear-solvers]
---

Every forward solve, adjoint solve and Jacobian of EIT is a sparse symmetric positive semidefinite system with several right-hand sides. This chapter covers direct and iterative solvers, preconditioners, and fast transform methods for structured domains.

**Reading order.**

1. Krylov methods: [[Conjugate Gradient Method]], [[Projected Conjugate Gradient]] for the singular Neumann problem, [[Block Conjugate Gradient]] for several patterns at once; [[MINRES]], [[LSQR]] and [[Block Krylov Methods]] for related problems.
2. Direct factorisation: [[Projected Cholesky Factorization]].
3. Preconditioning: [[Algebraic Multigrid]].
4. Fast transforms: [[Discrete Cosine Transform]], [[Fast Solvers on Rectangular Domains]], [[Fast Solvers on Disk Domains]].

A comparison with recommendations is [[Choosing a Linear Solver]].

**Related.** The null space and grounding are introduced in [[04 Finite Elements/index|Finite Elements]]; mapping general domains to the disk in [[06 Meshes and Geometry/index|Meshes and Geometry]].

## References

1. Y. Saad (2003). *Iterative Methods for Sparse Linear Systems*, 2nd ed. SIAM. [doi:10.1137/1.9780898718003](https://doi.org/10.1137/1.9780898718003)
