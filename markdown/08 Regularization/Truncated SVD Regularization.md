---
tags: [regularization, spectral]
aliases: [TSVD, Spectral truncation, SVD of the boundary operator]
---

**Truncated SVD (linear problems).** For $Ax=b$ with SVD $A = \sum_k s_k\,u_kv_k^\top$, the minimum-norm solution $x = \sum_k \frac{u_k^\top b}{s_k}v_k$ amplifies noise in the components with small $s_k$. Truncation keeps only the first $K$ terms:

$$
x_K = \sum_{k=1}^K \frac{u_k^\top b}{s_k}\,v_k .
$$

The truncation index $K$ plays the role of the [[Choosing the Regularization Parameter|regularisation parameter]].

**Spectral truncation of the EIT data.** The [[Neumann-to-Dirichlet Map]] is self-adjoint and positive definite on zero-mean functions (see [[Properties of the Boundary Operators]]). Given current patterns $G=[g_1,\dots,g_N]$ and measured voltages $F=[f_1,\dots,f_N]$ (zero-mean columns), estimate the discrete operator $\hat{\mathcal R} = F\,G^{+}$ (see [[Discrete Boundary Operator]]). Symmetrise it and take its eigen/singular value decomposition

$$
\hat{\mathcal R} = V\,\Sigma\,U^\top,\qquad \Sigma = \operatorname{diag}(s_1\ge s_2\ge\dots\ge0).
$$

This yields new, orthogonal data pairs ordered by importance:

$$
g_k^{\text{new}} = u_k,\qquad f_k^{\text{new}} = s_k\,v_k,\qquad k=1,\dots,K .
$$

- The $u_k$ are the current patterns with the largest response, which are the [[Current Patterns|optimal patterns]] in Isaacson's sense when the operator is the difference to a reference.
- Discarding pairs with $s_k$ below the noise level removes the components that carry mostly noise. It is regularisation *in data space* and needs no assumption on $\sigma$.
- The ratio $\sum_{k\le K}s_k^2/\sum_k s_k^2$ ("explained operator energy") or the operator-norm error $\|\hat{\mathcal R}-\hat{\mathcal R}_K\|_{\text{op}} = s_{K+1}$ helps choose $K$.

For proper function space weighting, the SVD should be computed in the discrete $L^2(\partial\Omega)$ geometry, that is, after transforming with $M_\Gamma^{1/2}$ (see [[Boundary Mass and Stiffness Matrices]]).

The singular values of the boundary operator decay (for a homogeneous disc like $1/k$). The singular values of the *linearised* map from conductivity to data decay much faster, which is the actual source of severe ill-posedness (see [[Decay of Boundary Measurements]]).

## Truncated SVD of the Jacobian

The same truncation applies to the conductivity, via the Jacobian $J$ of the residual $r(\sigma)$ (see [[Gauss-Newton Method]]). In a metric $W$ on the parameters, for example the lumped mass matrix of finite element coefficients, the SVD of $J W^{-1/2}$ gives

$$
J = U\,S\,V^\top W,\qquad V^\top W\,V = I,
$$

with parameter modes $v_i$ that are orthonormal in $W$. The modes are ordered by how well the measurements determine them. For EIT this ordering is by **depth**: the leading modes are concentrated near the boundary, and the later ones reach into the interior with rapidly decreasing singular values (see [[Linearized EIT and the Sensitivity Kernel]]). This is the geometry of the problem itself. The ordering depends on the electrodes, the current patterns and the domain, but not on an assumption about $\sigma$.

**Truncated Gauss–Newton.** The step uses only the leading $K$ modes of the current Jacobian, taking the minimum-norm step in $W$:

$$
\delta = -\sum_{i\le K}\frac{u_i^\top r}{s_i}\,v_i .
$$

Iterated until the misfit reaches the noise level (the discrepancy principle, see [[Choosing the Regularization Parameter]]), this is regularisation by projection with no penalty term. Too small a $K$ stalls above the noise level. Too large a $K$ fits the noise after a few iterations (semi-convergence). [[Levenberg-Marquardt Method|Levenberg–Marquardt]] with identity damping uses the same SVD but filters it smoothly, with factors $s_i^2/(s_i^2+\lambda)$ in place of the cut-off. Both respect the depth ordering. Smoothness and [[Total Variation]] penalties impose a different ordering, which the data do not see.

**Data-optimal subspace.** Computed once at a reference conductivity, the leading modes $V_K$ form a basis for a subspace parametrisation (see [[Parametrizations of the Conductivity]]). It is the data-adapted counterpart of low-frequency cosine modes.

**In ModularEIT.jl:** [`jacobian_svd`](https://danielboigk.github.io/ModularEIT.jl/dev/api/optimization/#ModularEIT.jacobian_svd), [`jacobian_basis`](https://danielboigk.github.io/ModularEIT.jl/dev/api/optimization/#ModularEIT.jacobian_basis), [`TruncatedGaussNewton`](https://danielboigk.github.io/ModularEIT.jl/dev/api/optimization/#ModularEIT.TruncatedGaussNewton).

## References

1. P. C. Hansen (1987). *The truncated SVD as a method for regularization*. BIT 27, 534–553. [doi:10.1007/BF01937276](https://doi.org/10.1007/BF01937276)
2. D. Isaacson (1986). *Distinguishability of Conductivities by Electric Current Computed Tomography*. IEEE Trans. Med. Imaging 5(2), 91–95. [doi:10.1109/TMI.1986.4307752](https://doi.org/10.1109/TMI.1986.4307752)
3. D. Gisser, D. Isaacson, J. C. Newell (1990). *Electric Current Computed Tomography and Eigenvalues*. SIAM J. Appl. Math. 50(6), 1623–1634. [doi:10.1137/0150096](https://doi.org/10.1137/0150096)
4. B. Kaltenbacher, A. Neubauer, O. Scherzer (2008). *Iterative Regularization Methods for Nonlinear Ill-Posed Problems*. de Gruyter. [doi:10.1515/9783110208276](https://doi.org/10.1515/9783110208276)
5. J. L. Mueller, S. Siltanen (2012). *Linear and Nonlinear Inverse Problems with Practical Applications*. SIAM. [doi:10.1137/1.9781611972344](https://doi.org/10.1137/1.9781611972344)
