---
tags: [regularization, reconstruction, fem]
aliases: [Pixel parametrization, Coarse-to-fine reconstruction, Regularization by parametrization]
---

The unknowns of a reconstruction need not be the coefficients of the conductivity in the finite element space of the forward model. A **parametrization** $\sigma = P\theta$ describes the conductivity by parameters $\theta$ of a different space. It separates two choices: the resolution needed to *simulate* the measurements, and the resolution at which the conductivity is *sought*.

## Chain rule

For a data objective $J(\sigma)$, the objective in the parameters is $\tilde J(\theta) = J(P\theta)$, with

$$
\nabla_\theta\tilde J = P^\top\nabla_\sigma J,\qquad \frac{\partial r}{\partial\theta} = \frac{\partial r}{\partial\sigma}\,P
$$

for the residual $r$ of a least-squares misfit. Adjoint gradients (see [[Adjoint State Method]]), [[Gauss-Newton Method|Gauss–Newton]] Jacobians and all optimisers carry over unchanged.

## Pixels on any mesh

With $\theta$ the values of an image on a rectangle containing the domain (see [[Pixel Images and Finite Element Functions]]), a natural $P$ is the lumped [[L2 Projection]] of the pixel function onto the conductivity space:

$$
P_{ak} = \frac{\int\varphi_a\,\chi_k\,\mathrm dx}{\int\varphi_a\,\mathrm dx},
$$

with $\chi_k$ the indicator of pixel $k$. The weights are non-negative and sum to one. Constants are therefore reproduced, every coefficient is a weighted average of pixel values, and bounds on the pixels bound the conductivity (see [[Box Constraints on Conductivity]]). On meshes whose cells lie inside single pixels the map is exact.

The forward mesh can then be refined where the physics demands it, for example towards the boundary and the electrodes (see [[Adaptive Meshing in EIT]]). The unknowns stay pixels: the natural variables of image priors (see [[Learned Regularization]]), and of pixel-based penalties such as [[Total Variation]].

## Subspaces: regularisation by parametrisation

Restricting the pixels to a subspace, $\theta_{\text{pixels}} = B\,c$, is itself a regulariser: only conductivities in the span of $B$ can be reconstructed (see [[Implicit Regularization]]).

- **Low-frequency cosine modes** (see [[Discrete Cosine Transform]]): a smooth, low-dimensional conductivity with few, well-determined unknowns. As the first stage of a **coarse-to-fine** reconstruction it provides a stable starting point for the full pixel problem.
- **Cosine modes plus a boundary band**: the measurements determine the conductivity near the boundary much better than in the interior (see [[Decay of Boundary Measurements]], [[Linearized EIT and the Sensitivity Kernel]]). A basis of smooth modes everywhere, plus free pixels in a band along the boundary, matches the resolution of the parametrisation to that of the data.
- **Singular vectors of the Jacobian**: the data-optimal subspace, ordered by how well each direction is determined (see [[Truncated SVD Regularization]]).

Subspace coefficients cannot be bounded like pixels, since a combination of modes can become negative. Trial conductivities have to be checked for positivity instead.

**In ModularEIT.jl:** [`PixelParametrization`](https://danielboigk.github.io/ModularEIT.jl/dev/api/parametrization/#ModularEIT.PixelParametrization), [`SubspaceParametrization`](https://danielboigk.github.io/ModularEIT.jl/dev/api/parametrization/#ModularEIT.SubspaceParametrization), [`ParametrizedObjective`](https://danielboigk.github.io/ModularEIT.jl/dev/api/parametrization/#ModularEIT.ParametrizedObjective), [`dct_basis`](https://danielboigk.github.io/ModularEIT.jl/dev/api/parametrization/#ModularEIT.dct_basis), [`boundary_band_basis`](https://danielboigk.github.io/ModularEIT.jl/dev/api/parametrization/#ModularEIT.boundary_band_basis), [`jacobian_basis`](https://danielboigk.github.io/ModularEIT.jl/dev/api/optimization/#ModularEIT.jacobian_basis).

## References

1. B. Kaltenbacher, A. Neubauer, O. Scherzer (2008). *Iterative Regularization Methods for Nonlinear Ill-Posed Problems*. de Gruyter. [doi:10.1515/9783110208276](https://doi.org/10.1515/9783110208276)
2. H. W. Engl, M. Hanke, A. Neubauer (1996). *Regularization of Inverse Problems*. Kluwer. [doi:10.1007/978-94-009-1740-8](https://doi.org/10.1007/978-94-009-1740-8)
3. J. L. Mueller, S. Siltanen (2012). *Linear and Nonlinear Inverse Problems with Practical Applications*. SIAM. [doi:10.1137/1.9781611972344](https://doi.org/10.1137/1.9781611972344)
