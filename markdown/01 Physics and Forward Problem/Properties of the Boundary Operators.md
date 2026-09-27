---
tags: [forward-problem, boundary-operator]
---

Let $u_f$ solve the [[Dirichlet Problem]] with data $f$. Green's first identity (see [[Green's Identities]]) applied to $\nabla\cdot(\gamma\nabla u_f)=0$ gives

$$
\langle \Lambda_\gamma f, h\rangle = \int_{\partial\Omega} h\,\gamma\partial_\nu u_f\,\mathrm ds = \int_\Omega \gamma\,\nabla u_f\cdot\nabla u_h\,\mathrm dx .
$$

Several properties of the [[Dirichlet-to-Neumann Map]] $\Lambda_\gamma$ follow from this symmetric form:

1. **Self-adjoint:** $\langle\Lambda_\gamma f,h\rangle = \langle f,\Lambda_\gamma h\rangle$.
2. **Positive semidefinite:** $\langle\Lambda_\gamma f,f\rangle = \int_\Omega\gamma|\nabla u_f|^2 \ge 0$, which is the dissipated power.
3. **Kernel = constants:** equality holds iff $\nabla u_f = 0$, that is, $f$ is constant.
4. **Zero-mean range:** taking $h = 1$ gives $\int_{\partial\Omega}\Lambda_\gamma f = 0$.
5. **Homogeneity:** $\Lambda_{c\gamma} = c\,\Lambda_\gamma$ for constants $c>0$.
6. **Monotonicity:** $\gamma_1\le\gamma_2$ a.e. implies $\Lambda_{\gamma_1}\le\Lambda_{\gamma_2}$ as quadratic forms. This follows from the [[Dirichlet and Thomson Principles|Dirichlet principle]] and is the basis of monotonicity-based inclusion detection.
7. **Nonlinearity:** in general $\Lambda_{\gamma_1+\gamma_2}\ne\Lambda_{\gamma_1}+\Lambda_{\gamma_2}$ (see [[Forward Map]]).

The [[Neumann-to-Dirichlet Map]] inherits the corresponding properties on zero-mean functions, with the monotonicity reversed: $\gamma_1\le\gamma_2 \Rightarrow \mathcal R_{\gamma_1}\ge\mathcal R_{\gamma_2}$.

Because $\mathcal R_\gamma$ is self-adjoint and positive, its discretisation has an orthogonal eigendecomposition $\mathcal R = Q\Sigma Q^\top$ with $\Sigma\ge0$. This is used for [[Truncated SVD Regularization]] and for choosing [[Current Patterns]].

## References

1. L. Borcea (2002). *Electrical impedance tomography*. Inverse Problems 18(6), R99–R136. [doi:10.1088/0266-5611/18/6/201](https://doi.org/10.1088/0266-5611/18/6/201)
2. D. Gisser, D. Isaacson, J. C. Newell (1990). *Electric Current Computed Tomography and Eigenvalues*. SIAM J. Appl. Math. 50(6), 1623–1634. [doi:10.1137/0150096](https://doi.org/10.1137/0150096)
3. B. Harrach, M. Ullrich (2013). *Monotonicity-based shape reconstruction in electrical impedance tomography*. SIAM J. Math. Anal. 45(6), 3382–3403. [doi:10.1137/120886984](https://doi.org/10.1137/120886984)
