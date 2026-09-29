---
tags: [overview, linear-solvers]
---

The systems $A(\sigma)X = B$ of EIT are sparse, symmetric and positive semidefinite. The current-driven problem is singular (the constants), and there are several right-hand sides. They are solved many times with slowly changing σ. Three families of solvers fit, with different trade-offs.

## Direct factorisation

A sparse Cholesky factorisation, adapted to the singular Neumann matrix (see [[Projected Cholesky Factorization]]), solves all right-hand sides at the cost of one factorisation and cheap triangular solves.

- **Strengths.** Robust, independent of the conductivity contrast, and very fast in 2D. The symbolic analysis (the fill-reducing ordering) is reused when only σ changes.
- **Limits.** Fill-in and memory grow quickly in 3D and on very fine meshes.

## Preconditioned Krylov methods

The [[Conjugate Gradient Method]], restricted to the complement of the null space (see [[Projected Conjugate Gradient]]), needs only matrix–vector products. [[Block Conjugate Gradient]] treats all patterns at once, and [[MINRES]] and [[LSQR]] cover the indefinite and least-squares variants (see [[Block Krylov Methods]]). The iteration count depends on the preconditioner:

- **Jacobi.** Cheap, but the iterations grow like $1/h$.
- **[[Algebraic Multigrid]].** Close to mesh-independent on well-shaped meshes. It degrades on strongly anisotropic meshes and needs a setup for every σ.
- **Fast transforms.** On structured meshes, the constant-coefficient problem is inverted exactly by FFT-type transforms, and the iteration count depends only on the contrast $\sigma_{\max}/\sigma_{\min}$ (see below).

A warm start from the previous solution helps when σ changes little between iterations.

## Fast transform solvers

- On uniform rectangle grids, for example pixel images, the [[Discrete Cosine Transform]] diagonalises the constant-coefficient operator (see [[Fast Solvers on Rectangular Domains]]).
- On rotationally symmetric disk meshes, an FFT in the angle decouples the problem into radial tridiagonal systems (see [[Fast Solvers on Disk Domains]]).
- Other simply connected domains are reached by a conformal map of the disk (see [[Numerical Conformal Mapping]]).

As preconditioners they are independent of the mesh and need no σ-dependent setup. They cover every electrode model through low-rank corrections.

## Rule of thumb

- **Unstructured 2D meshes:** direct factorisation.
- **Disks, rectangles or conformal images of the disk:** CG with the fast transform preconditioner, especially for many right-hand sides or on GPUs.
- **Large 3D problems:** CG with algebraic multigrid.

In every case the null space and the grounding must be handled consistently (see [[Null Space of the Neumann Problem]], [[Grounding of the Potential]]).

## References

1. Y. Saad (2003). *Iterative Methods for Sparse Linear Systems*, 2nd ed. SIAM. [doi:10.1137/1.9780898718003](https://doi.org/10.1137/1.9780898718003)
2. P. Concus, G. H. Golub (1973). *Use of Fast Direct Methods for the Efficient Numerical Solution of Nonseparable Elliptic Equations*. SIAM J. Numer. Anal. 10(6), 1103–1120. [doi:10.1137/0710092](https://doi.org/10.1137/0710092)
