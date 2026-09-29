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
- $\operatorname{diag}(J^\top J)$ (Marquardt 1963, scale-invariant; in EIT the NOSER prior of Cheney et al. 1990);
- $\operatorname{diag}\lVert J e_j\rVert$, the sensitivities: the geometric mean of the two previous choices (see below);
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

**Damping in EIT.** The sensitivities $\lVert J e_j\rVert$ of EIT span orders of magnitude. They are largest next to the electrode edges, where the current density is singular, and smallest in the interior (see [[Decay of Boundary Measurements]]). With identity damping every parameter is damped equally, so the steps concentrate on the most sensitive parameters. The result is a periodic artefact along the boundary, with the period of the electrodes. Marquardt's $\operatorname{diag}(J^\top J)$ over-corrects: it lets the insensitive interior parameters move as freely as the boundary ones, and amplifies noise there. Damping with $\operatorname{diag}\lVert J e_j\rVert$ lies between the two. In reconstructions of images with 32 electrodes it removes the electrode artefacts and gives clearly the smallest error of the three.

**Trust-region view.** Adapting $\lambda$ from the ratio of actual to predicted reduction makes LM a trust-region method: increase $\lambda$ after a poor step and decrease it after a good one.

**In ModularEIT.jl:** [`GaussNewton`](https://danielboigk.github.io/ModularEIT.jl/dev/api/optimization/#ModularEIT.GaussNewton).

## References

1. K. Levenberg (1944). *A method for the solution of certain non-linear problems in least squares*. Quart. Appl. Math. 2(2), 164–168. [doi:10.1090/qam/10666](https://doi.org/10.1090/qam/10666)
2. D. W. Marquardt (1963). *An Algorithm for Least-Squares Estimation of Nonlinear Parameters*. J. SIAM 11(2), 431–441. [doi:10.1137/0111030](https://doi.org/10.1137/0111030)
3. M. Hanke (1997). *A regularizing Levenberg–Marquardt scheme, with applications to inverse groundwater filtration problems*. Inverse Problems 13(1), 79–95. [doi:10.1088/0266-5611/13/1/007](https://doi.org/10.1088/0266-5611/13/1/007)
4. M. Cheney, D. Isaacson, J. C. Newell, S. Simske, J. Goble (1990). *NOSER: An algorithm for solving the inverse conductivity problem*. Int. J. Imaging Syst. Technol. 2(2), 66–75. [doi:10.1002/ima.1850020203](https://doi.org/10.1002/ima.1850020203)
5. J. Nocedal, S. J. Wright (2006). *Numerical Optimization*, 2nd ed. Springer. [doi:10.1007/978-0-387-40065-5](https://doi.org/10.1007/978-0-387-40065-5)
