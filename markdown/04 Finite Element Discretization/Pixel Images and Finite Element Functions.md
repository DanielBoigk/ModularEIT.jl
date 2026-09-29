---
tags: [numerics, fem, images]
aliases: [Image to mesh, Mesh to image, Rasterisation]
---

Learned priors, image denoisers and plots work with pixel images, while the forward problem works with finite element coefficients. Two linear maps connect them. Let the image have $n\times m$ pixels with centres $x_{ij}$, and let the finite element space have basis functions $\psi_a$ (conductivity) or $\varphi_i$ (potential).

**Mesh → image (evaluation).** Evaluate the finite element function at the pixel centres:

$$
\mathrm{img}_{ij} = \sum_a \sigma_a\,\psi_a(x_{ij}),\qquad \mathrm{vec}(\mathrm{img}) = S\,\boldsymbol\sigma .
$$

$S$ is sparse, with at most as many nonzeros per row as basis functions per cell, and depends only on the mesh and the image size. It can be assembled once. Pixels whose centres lie outside a non-rectangular domain receive a fill value. The transpose $S^\top$ maps gradients in image space, for example of a learned regulariser, back to coefficient space.

**Image → mesh.** Two natural choices:

- *Sampling (interpolation).* Evaluate the image, interpolated bilinearly between pixel centres, at the nodes of the finite element space; for piecewise constants these are the cell centroids. Bilinear interpolation reproduces linear functions exactly, so linear conductivities survive a round trip on any mesh. Sampling ignores the pixels between the nodes, so fine images alias on coarse meshes.
- *[[L2 Projection]].* Treat the image as a function that is constant on each pixel, and solve $M\boldsymbol\sigma = \big(\int \mathrm{img}\,\psi_a\big)_a$. This averages the pixels over each cell and preserves the integral. It is the right choice when the image is finer than the mesh. In practice the integrals are computed by quadrature, and the rule must place several points in every pixel.

**Pixel-aligned meshes.** If the mesh is the pixel grid itself (one quadrilateral per pixel, piecewise constant $\sigma$), $S$ is a permutation and both directions are exact inverses: images and coefficient vectors are the same data in a different order. This is the natural discretisation for image-based EIT with image priors such as [[Diffusion Models for EIT]]. [[Hanging Nodes|Adaptive refinement]] breaks this one-to-one correspondence. The maps above then connect the refined mesh to a fixed image grid.

**Orientation.** Images are usually stored row by row from the *top*, while coordinates grow upwards. The pixel in row $i$ and column $j$ of an $n\times m$ image of the box $[x_0,x_1]\times[y_0,y_1]$ has the centre $\big(x_0+(j-\tfrac12)\Delta x,\ y_1-(i-\tfrac12)\Delta y\big)$.

On pixel-aligned rectangle meshes the finite element matrices are diagonalised by the [[Discrete Cosine Transform]], which gives fast solvers and preconditioners ([[Fast Solvers on Rectangular Domains]]) and spectral norms ([[Spectral Sobolev Norms on Rectangles]]).

**In ModularEIT.jl:** [`ImageMap`](https://danielboigk.github.io/ModularEIT.jl/dev/api/images/#ModularEIT.ImageMap), [`to_image`](https://danielboigk.github.io/ModularEIT.jl/dev/api/images/#ModularEIT.to_image), [`from_image`](https://danielboigk.github.io/ModularEIT.jl/dev/api/images/#ModularEIT.from_image).

## References

1. S. C. Brenner, L. R. Scott (2008). *The Mathematical Theory of Finite Element Methods*, 3rd ed. Springer. [doi:10.1007/978-0-387-75934-0](https://doi.org/10.1007/978-0-387-75934-0)
