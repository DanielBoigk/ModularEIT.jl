---
tags: [inverse-problem, foundations]
aliases: [Ill-posedness, Hadamard well-posedness, Ill-posed problem]
---

Following Hadamard (1902; see Engl, Hanke & Neubauer, Ch. 1), a problem $F(x) = y$ is **well-posed** if

1. a solution exists for every admissible $y$ (*existence*);
2. the solution is unique (*uniqueness*);
3. the solution depends continuously on $y$ (*stability*).

A problem that violates any of these is **ill-posed**. Inverse problems are typically ill-posed because the forward operator $F$ smooths: it is compact, or at least its inverse is unbounded.

For EIT:

- *Existence* fails for noisy data. A measured noisy operator is generally not the DtN map of any conductivity.
- *Uniqueness* holds for isotropic conductivities under mild regularity assumptions (see [[Uniqueness in the Calderón Problem]]). It fails for anisotropic ones (see [[Anisotropic Conductivities]]).
- *Stability* is only logarithmic (see [[Stability of the Calderón Problem]]). EIT is therefore called **severely** (exponentially) ill-posed. Mildly ill-posed problems, such as the Radon transform of CT, have algebraic decay of singular values.

For linear problems, the degree of ill-posedness is measured by the decay rate of the singular values of $F$. Unless the problem is restricted to a finite-dimensional or compact set, **regularization** is needed to obtain stable approximate solutions (see [[Variational Regularization]]).

## References

1. H. W. Engl, M. Hanke, A. Neubauer (1996). *Regularization of Inverse Problems*. Kluwer. [doi:10.1007/978-94-009-1740-8](https://doi.org/10.1007/978-94-009-1740-8)
2. J. L. Mueller, S. Siltanen (2012). *Linear and Nonlinear Inverse Problems with Practical Applications*. SIAM. [doi:10.1137/1.9781611972344](https://doi.org/10.1137/1.9781611972344)
