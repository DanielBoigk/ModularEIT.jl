---
tags: [numerics, fem, adjoint, gpu]
aliases: [Tensor assembler, Stiffness tensor]
---

The [[Weighted Stiffness Matrix]] is **linear in the conductivity**. With $\sigma = \sum_a\sigma_a\psi_a$ in its own finite element space,

$$
L(\sigma) = \sum_a \sigma_a L_a,\qquad (L_a)_{ij} = \int_\Omega \psi_a\,\nabla\varphi_i\cdot\nabla\varphi_j\,\mathrm dx .
$$

The sparsity pattern of $L$ does not depend on $\sigma$. Number the stored entries $k = 1,\dots,\mathrm{nnz}(L)$, with row $r_k$ and column $c_k$. Then the stored values are one matrix-vector product:

$$
\operatorname{nzval}\big(L(\sigma)\big) = T\,\sigma,\qquad T_{ka} = (L_a)_{r_k c_k}.
$$

$T$ is sparse, of size $\mathrm{nnz}(L)\times n_\sigma$, and built once per mesh by the usual cell loop: cell $K$ contributes the local tensor $\int_K\psi_a\nabla\varphi_i\cdot\nabla\varphi_j$ (see [[Numerical Quadrature and Assembly]]). The quadrature must integrate a polynomial of degree $2(p_u-1)+p_\sigma$ exactly, where $p_u$, $p_\sigma$ are the polynomial degrees of the two spaces.

## Gradients from the same tensor

Every derivative of a bilinear expression in $L$ is a contraction with $T$:

$$
\frac{\partial}{\partial\sigma_a}\big(\boldsymbol\lambda^\top L(\sigma)\,\mathbf u\big) = \boldsymbol\lambda^\top L_a\mathbf u = \sum_k T_{ka}\,\lambda_{r_k}u_{c_k} = (T^\top\mathbf w)_a,\qquad w_k = \lambda_{r_k}u_{c_k}.
$$

This is exactly $\int_\Omega\psi_a\nabla u_h\cdot\nabla\lambda_h\,\mathrm dx$, the discrete gradient of the [[Adjoint State Method]]. It is exact because assembly and gradient use the same quadrature (see [[Discretize-then-Optimize vs Optimize-then-Discretize]]). Sums over current patterns only change $\mathbf w$: $w_k = \sum_s\lambda_{s,r_k}u_{s,c_k}$.

- **Adjoint state:** $\nabla J = -T^\top\mathbf w(\boldsymbol\lambda,\mathbf u)$.
- **Voltage-driven data:** $+T^\top\mathbf w$ (see [[Adjoint Method for the Dirichlet Problem]]).
- **[[Kohn-Vogelius Functional]]:** $\tfrac12 T^\top\big(\mathbf w(\mathbf u_D,\mathbf u_D)-\mathbf w(\mathbf u_N,\mathbf u_N)\big)$, with no adjoint at all.
- **Jacobian rows:** $T^\top W$ with $W_{kj} = z_{j,r_k}u_{c_k}$ for a block of adjoint fields $z_j$, i.e. one sparse times dense product per pattern.

## L² gradient

The vector $T^\top\mathbf w$ is a *dual* vector: its entries are integrals against the basis functions $\psi_a$, so they scale with the cell sizes. The [[L2 Projection]] of the continuous gradient density $-\nabla u\cdot\nabla\lambda$ onto the $\sigma$ space solves

$$
M_\sigma\,\mathbf g_{L^2} = -T^\top\mathbf w ,
$$

with the mass matrix $M_\sigma$ of the $\sigma$ space. This is the Riesz representative in $L^2$ (see [[Gradient Representation and the Riesz Map]]). It is mesh independent and usually a better search direction on graded meshes. For piecewise constant $\sigma$, $M_\sigma$ is diagonal with the cell areas, so $\mathbf g_{L^2}$ is the coefficient gradient divided by the cell areas. Both gradients come from the same contraction and differ only by the Riesz map.

## Parallelism

Assembly ($T\sigma$), gradient ($T^\top\mathbf w$) and the gather $\mathbf w$ are sparse matrix-vector products and one independent product per stored entry. They parallelise over threads and GPU cores without atomic operations or colouring. The memory cost is $\mathrm{nnz}(T)\approx n_{\mathrm{cells}}\,n_u^2\,n_\psi$ for $n_u$ potential and $n_\psi$ conductivity basis functions per cell. That is small in 2D, and needs checking in 3D with high-order conductivity spaces.

## References

1. S. C. Brenner, L. R. Scott (2008). *The Mathematical Theory of Finite Element Methods*, 3rd ed. Springer. [doi:10.1007/978-0-387-75934-0](https://doi.org/10.1007/978-0-387-75934-0)
2. M. Hinze, R. Pinnau, M. Ulbrich, S. Ulbrich (2009). *Optimization with PDE Constraints*. Springer. [doi:10.1007/978-1-4020-8839-1](https://doi.org/10.1007/978-1-4020-8839-1)
3. N. Polydorides, W. R. B. Lionheart (2002). *A Matlab toolkit for three-dimensional electrical impedance tomography: a contribution to the Electrical Impedance and Diffuse Optical Reconstruction Software project*. Meas. Sci. Technol. 13(12), 1871–1883. [doi:10.1088/0957-0233/13/12/310](https://doi.org/10.1088/0957-0233/13/12/310)
