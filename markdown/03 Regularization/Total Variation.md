---
tags: [regularization, classical]
aliases: [TV, TV regularization]
---

The **total variation** of a function $z$ on $\Omega$ is

$$
\operatorname{TV}(z) = \int_\Omega |\nabla z|\,\mathrm dx
\qquad\Big(\text{generally } \sup\Big\{\int_\Omega z\,\nabla\cdot\mathbf p\ :\ \mathbf p\in C_c^1(\Omega)^n,\ |\mathbf p|\le1\Big\}\Big).
$$

The supremum form also makes sense for discontinuous $z$. For a piecewise constant function, TV equals the jump height times the length of the jump set. TV therefore favours **piecewise constant** images with sharp edges, which fits organs or inclusions in a background. It was introduced for image denoising by Rudin, Osher and Fatemi (ROF, 1992).

**Isotropic vs. anisotropic.** $|\nabla z| = \sqrt{z_x^2+z_y^2}$ is isotropic. $|z_x|+|z_y|$ is the anisotropic variant: cheaper and separable, but it prefers axis-aligned edges.

**Non-differentiability.** $|\nabla z|$ is not differentiable where $\nabla z = 0$. There are two standard remedies:

1. **Smoothing:** replace $|\cdot|$ by a differentiable approximation and use gradient-based methods (see [[Smoothed Total Variation]]).
2. **Primal–dual / splitting:** use the dual form above and solve with the Chambolle–Pock algorithm or [[ADMM]], which handle the non-smooth term exactly.

**Proximal operator (ROF problem).**

$$
\operatorname{prox}_{\beta\mathrm{TV}}(y) = \arg\min_z\ \beta\operatorname{TV}(z) + \tfrac\rho2\|z-y\|_2^2 .
$$

This is TV denoising itself. It is used as the regulariser step in [[ADMM]].

In EIT, TV regularisation gives sharper inclusions than [[Tikhonov Regularization]] and is robust in clinical data (Borsic et al. 2010).

## References

1. L. I. Rudin, S. Osher, E. Fatemi (1992). *Nonlinear total variation based noise removal algorithms*. Physica D 60(1–4), 259–268. [doi:10.1016/0167-2789(92)90242-F](https://doi.org/10.1016/0167-2789(92)90242-F)
2. A. Chambolle, V. Caselles, D. Cremers, M. Novaga, T. Pock (2010). *An Introduction to Total Variation for Image Analysis*. In: Theoretical Foundations and Numerical Methods for Sparse Recovery, de Gruyter, 263–340. [doi:10.1515/9783110226157.263](https://doi.org/10.1515/9783110226157.263)
3. A. Chambolle, T. Pock (2011). *A First-Order Primal-Dual Algorithm for Convex Problems with Applications to Imaging*. J. Math. Imaging Vis. 40, 120–145. [doi:10.1007/s10851-010-0251-1](https://doi.org/10.1007/s10851-010-0251-1)
4. A. Borsic, B. M. Graham, A. Adler, W. R. B. Lionheart (2010). *In Vivo Impedance Imaging With Total Variation Regularization*. IEEE Trans. Med. Imaging 29(1), 44–54. [doi:10.1109/TMI.2009.2022540](https://doi.org/10.1109/TMI.2009.2022540)
5. Y. Wang (2022). *Anisotropic TV Regularization in Electrical Impedance Tomography: An Experimental Study*. Engineering 14(3), 138–146. [doi:10.4236/eng.2022.143013](https://doi.org/10.4236/eng.2022.143013)
