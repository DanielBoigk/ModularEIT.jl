---
tags: [numerics, linear-algebra]
---

**LSQR** (Paige & Saunders 1982) solves least-squares problems

$$
\min_{\mathbf x}\|A\mathbf x-\mathbf b\|_2\qquad\text{or}\qquad\min_{\mathbf x}\left\|\begin{pmatrix}A\\ \sqrt\lambda\,B\end{pmatrix}\mathbf x-\begin{pmatrix}\mathbf b\\ 0\end{pmatrix}\right\|_2
$$

for rectangular, sparse or matrix-free $A$. It uses only products with $A$ and $A^\top$. In exact arithmetic it is equivalent to CG applied to the normal equations $A^\top A\mathbf x = A^\top\mathbf b$, but it is numerically more stable because it never forms $A^\top A$, whose condition number is the *square* of that of $A$. It is based on Golub–Kahan bidiagonalisation.

**In EIT.** A [[Levenberg-Marquardt Method|Levenberg–Marquardt]] step minimises $\|J\delta+r\|^2+\lambda\|B\delta\|^2$, where $B^\top B = L_{\text{LM}}$ is the damping matrix. This is exactly the stacked least-squares problem above, with $A = J$ and $\mathbf b = -r$. With $B=I$ it is plain damped least squares, which LSQR supports directly through its damping parameter. For a general $L_{\text{LM}}$, use a factor $B$ (for example the Cholesky factor of the mass or stiffness matrix) or change variables.

Early termination of LSQR acts as an additional [[Implicit Regularization|implicit regularisation]].

## References

1. C. C. Paige, M. A. Saunders (1982). *LSQR: An Algorithm for Sparse Linear Equations and Sparse Least Squares*. ACM Trans. Math. Softw. 8(1), 43–71. [doi:10.1145/355984.355989](https://doi.org/10.1145/355984.355989)
2. Å. Björck (1996). *Numerical Methods for Least Squares Problems*. SIAM. [doi:10.1137/1.9781611971484](https://doi.org/10.1137/1.9781611971484)
