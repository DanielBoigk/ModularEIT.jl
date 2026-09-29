---
tags: [regularization, classical]
aliases: [Huber TV, Charbonnier TV, Differentiable TV]
---

To make [[Total Variation]] differentiable, the Euclidean norm $|\nabla z|$ is replaced by a smooth approximation $\phi_\varepsilon(|\nabla z|)$ with a small $\varepsilon>0$:

**Charbonnier (pseudo-Huber) smoothing**

$$
\operatorname{TV}_\varepsilon(z) = \int_\Omega \sqrt{|\nabla z|^2+\varepsilon^2}\,\mathrm dx .
$$

**Huber smoothing** (quadratic near zero, linear far away)

$$
\phi_\varepsilon(t) = \begin{cases} \dfrac{t^2}{2\varepsilon}, & t\le\varepsilon,\\[4pt] t-\dfrac\varepsilon2, & t>\varepsilon,\end{cases}
\qquad \operatorname{TV}^{\text{Hub}}_\varepsilon(z) = \int_\Omega\phi_\varepsilon(|\nabla z|)\,\mathrm dx .
$$

Both converge to TV as $\varepsilon\to0$. Huber is exactly linear above $\varepsilon$ and preserves edges slightly better. Charbonnier is $C^\infty$.

**Gradient.** The first variation of $\operatorname{TV}_\varepsilon$ is, with natural boundary conditions,

$$
\nabla\operatorname{TV}_\varepsilon(z) = -\nabla\cdot\left(\frac{\nabla z}{\sqrt{|\nabla z|^2+\varepsilon^2}}\right).
$$

In a finite element space this is assembled as the vector $\big(\int_\Omega \frac{\nabla z\cdot\nabla\varphi_i}{\sqrt{|\nabla z|^2+\varepsilon^2}}\,\mathrm dx\big)_i$. No second derivatives of $z$ are needed.

**Hessian.** The Hessian has the structure of a weighted stiffness matrix with a rank-one correction. It is sparse but becomes very ill-conditioned as $\varepsilon\to0$. A common alternative is the *lagged diffusivity* fixed-point iteration (Vogel & Oman), which freezes the weight $1/\sqrt{|\nabla z|^2+\varepsilon^2}$ at the previous iterate. Quasi-Newton methods such as [[L-BFGS]] avoid forming the Hessian altogether.

**Choosing $\varepsilon$.** If $\varepsilon$ is too large, the result looks like $H^1$-[[Tikhonov Regularization]] and edges blur. If it is too small, the problem becomes stiff and optimisers slow down. Continuation, decreasing $\varepsilon$ during the iterations, is common.

**In ModularEIT.jl:** [`TotalVariationRegularizer`](https://danielboigk.github.io/ModularEIT.jl/dev/api/optimization/#ModularEIT.TotalVariationRegularizer), [`total_variation`](https://danielboigk.github.io/ModularEIT.jl/dev/api/discretization/#ModularEIT.total_variation).

## References

1. P. Charbonnier, L. Blanc-Féraud, G. Aubert, M. Barlaud (1997). *Deterministic edge-preserving regularization in computed imaging*. IEEE Trans. Image Process. 6(2), 298–311. [doi:10.1109/83.551699](https://doi.org/10.1109/83.551699)
2. P. J. Huber (1964). *Robust Estimation of a Location Parameter*. Ann. Math. Statist. 35(1), 73–101. [doi:10.1214/aoms/1177703732](https://doi.org/10.1214/aoms/1177703732)
3. C. R. Vogel, M. E. Oman (1996). *Iterative Methods for Total Variation Denoising*. SIAM J. Sci. Comput. 17(1), 227–238. [doi:10.1137/0917016](https://doi.org/10.1137/0917016)
4. A. Borsic, B. M. Graham, A. Adler, W. R. B. Lionheart (2010). *In Vivo Impedance Imaging With Total Variation Regularization*. IEEE Trans. Med. Imaging 29(1), 44–54. [doi:10.1109/TMI.2009.2022540](https://doi.org/10.1109/TMI.2009.2022540)
