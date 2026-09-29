---
tags: [optimization, regularization]
aliases: [Levenberg-Marquardt, LM, Damped Gauss-Newton]
---

The **Levenberg–Marquardt (LM)** method damps each [[Gauss-Newton Method|Gauss–Newton]] step:

$$
\big(J^\top J+\lambda\,L_{\text{LM}}\big)\,\delta = -J^\top r ,
$$

with a damping parameter $\lambda>0$ and a symmetric positive definite matrix $L_{\text{LM}}$. Common choices are:

- $I$ (classical LM; Levenberg 1944);
- $\operatorname{diag}(J^\top J)$ (Marquardt 1963, scale-invariant);
- the [[Mass Matrix]] $M$ (discretisation-independent $L^2$ damping);
- the [[Stiffness Matrix]] $K$ ($H^1$-smoothness of the step; add a small multiple of $M$ to make it definite).

**Equivalent least-squares form.** With a factor $B$, $B^\top B = L_{\text{LM}}$ (e.g. Cholesky), the step minimises

$$
\|r+J\delta\|^2+\lambda\|B\delta\|^2
\quad\Longleftrightarrow\quad
\min_\delta\left\|\begin{pmatrix}J\\ \sqrt\lambda\,B\end{pmatrix}\delta-\begin{pmatrix}-r\\ 0\end{pmatrix}\right\|^2 .
$$

This is a rectangular, non-symmetric system. It is best solved by [[LSQR]]. The normal-equations form is SPD and suits [[Conjugate Gradient Method|CG]].

**Interpretation.**

- As $\lambda\to0$: the Gauss–Newton step.
- As $\lambda\to\infty$: a short steepest-descent step, $\delta\approx-\lambda^{-1}L_{\text{LM}}^{-1}J^\top r$.
- Each step is a Tikhonov-regularised linearised problem (see [[Tikhonov Regularization]]). The regularisation acts on the *update*, not on the solution.

**Regularising LM (Hanke 1997).** For ill-posed problems, choose $\lambda_k$ at each step so that the linearised residual satisfies $\|r+J\delta\| = q\,\|r\|$ with $q<1$, and stop by the discrepancy principle. The iteration is then a convergent regularisation method.

**Trust-region view.** Adapting $\lambda$ from the ratio of actual to predicted reduction makes LM a trust-region method: increase $\lambda$ after a poor step and decrease it after a good one.

**In ModularEIT.jl:** [`GaussNewton`](https://danielboigk.github.io/ModularEIT.jl/dev/api/optimization/#ModularEIT.GaussNewton).

## References

1. K. Levenberg (1944). *A method for the solution of certain non-linear problems in least squares*. Quart. Appl. Math. 2(2), 164–168. [doi:10.1090/qam/10666](https://doi.org/10.1090/qam/10666)
2. D. W. Marquardt (1963). *An Algorithm for Least-Squares Estimation of Nonlinear Parameters*. J. SIAM 11(2), 431–441. [doi:10.1137/0111030](https://doi.org/10.1137/0111030)
3. M. Hanke (1997). *A regularizing Levenberg–Marquardt scheme, with applications to inverse groundwater filtration problems*. Inverse Problems 13(1), 79–95. [doi:10.1088/0266-5611/13/1/007](https://doi.org/10.1088/0266-5611/13/1/007)
4. J. Nocedal, S. J. Wright (2006). *Numerical Optimization*, 2nd ed. Springer. [doi:10.1007/978-0-387-40065-5](https://doi.org/10.1007/978-0-387-40065-5)
