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

## References

1. P. C. Hansen (1987). *The truncated SVD as a method for regularization*. BIT 27, 534–553. [doi:10.1007/BF01937276](https://doi.org/10.1007/BF01937276)
2. D. Isaacson (1986). *Distinguishability of Conductivities by Electric Current Computed Tomography*. IEEE Trans. Med. Imaging 5(2), 91–95. [doi:10.1109/TMI.1986.4307752](https://doi.org/10.1109/TMI.1986.4307752)
3. D. Gisser, D. Isaacson, J. C. Newell (1990). *Electric Current Computed Tomography and Eigenvalues*. SIAM J. Appl. Math. 50(6), 1623–1634. [doi:10.1137/0150096](https://doi.org/10.1137/0150096)
4. J. L. Mueller, S. Siltanen (2012). *Linear and Nonlinear Inverse Problems with Practical Applications*. SIAM. [doi:10.1137/1.9781611972344](https://doi.org/10.1137/1.9781611972344)
