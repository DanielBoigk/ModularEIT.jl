---
tags: [data, methodology]
---

An **inverse crime** is committed when synthetic data are generated with the *same* numerical model that is later used for the reconstruction: same mesh, same element type, same discretisation of $\sigma$, no noise. Discretisation errors then cancel exactly, and reconstructions look far better than they would on real data.

**How to avoid it.**

- Simulate data on a **finer or different mesh** (and element degree) than the reconstruction mesh, and interpolate the true conductivity independently.
- Represent the true conductivity in a way the reconstruction basis cannot represent exactly, for example with smooth-boundaried inclusions on a pixel grid.
- Always add realistic **noise** (see [[Noise Models for EIT Data]]).
- Ideally test on phantom tank or clinical data, where the forward model is never exact.

The concept was named by Colton and Kress and is emphasised in the EIT literature by Mueller and Siltanen, and by Kaipio and Somersalo.

## References

1. D. Colton, R. Kress (2013). *Inverse Acoustic and Electromagnetic Scattering Theory*, 3rd ed. Springer. [doi:10.1007/978-1-4614-4942-3](https://doi.org/10.1007/978-1-4614-4942-3)
2. J. Kaipio, E. Somersalo (2007). *Statistical inverse problems: Discretization, model reduction and inverse crimes*. J. Comput. Appl. Math. 198(2), 493–504. [doi:10.1016/j.cam.2005.09.027](https://doi.org/10.1016/j.cam.2005.09.027)
3. J. L. Mueller, S. Siltanen (2012). *Linear and Nonlinear Inverse Problems with Practical Applications*. SIAM. [doi:10.1137/1.9781611972344](https://doi.org/10.1137/1.9781611972344)
