---
tags: [data]
aliases: [Training data, Image prior data]
---

Learned priors (see [[Learned Regularization]]) need samples from a distribution of plausible conductivities. Possible sources:

1. **Real EIT reconstructions or anatomical atlases** (thorax, head, breast). These are closest to the application, but scarce, and they carry the blur of the reconstruction method that produced them.
2. **Parametric phantoms**: random ellipses, circles, polygons, or smooth random fields such as Gaussian processes. They are cheap and controllable but lack realistic texture.
3. **Natural grayscale images** (for example Tiny ImageNet, landscape collections) resampled onto the conductivity grid. They are rich in edges, textures and multi-scale structure, which makes them a demanding stress test of whether a method can recover sharp features. They are not anatomical, however.

**Mapping images to conductivities.** Intensities in $[0,1]$ are mapped affinely to $[\sigma_{\min},\sigma_{\max}]$ with $\sigma_{\min}>0$, so the forward problem stays elliptic (see [[Box Constraints on Conductivity]]). They are then interpolated onto the finite element space of $\sigma$. On an $n\times n$ grid of $Q_1$ elements, nodal values correspond directly to pixels.

**Held-out evaluation.** Test conductivities must not appear in training. Test data should also be simulated on a different, preferably finer, mesh than the reconstruction mesh to avoid the [[Inverse Crime]].

## References

1. Tiny ImageNet (2017). Kaggle dataset. [kaggle.com/c/tiny-imagenet](https://kaggle.com/competitions/tiny-imagenet)
2. S. Arridge, P. Maass, O. Öktem, C.-B. Schönlieb (2019). *Solving inverse problems using data-driven models*. Acta Numerica 28, 1–174. [doi:10.1017/S0962492919000059](https://doi.org/10.1017/S0962492919000059)
3. J. L. Mueller, S. Siltanen (2012). *Linear and Nonlinear Inverse Problems with Practical Applications*. SIAM. [doi:10.1137/1.9781611972344](https://doi.org/10.1137/1.9781611972344)
