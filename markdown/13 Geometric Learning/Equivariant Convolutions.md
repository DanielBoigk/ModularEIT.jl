---
tags: [geometric-learning, architecture]
aliases: [Group convolution, Steerable CNN, G-CNN]
---

Ordinary convolutions are translation-equivariant. **Group-equivariant convolutions** extend this to rotations and reflections.

**Lifting and group convolutions (G-CNN; Cohen & Welling 2016).** For a finite group $G$ such as the [[Dihedral Group D4]]:

1. *Lifting layer:* convolve the input with all 8 transformed copies $\rho(g)w$ of each filter. This produces feature maps indexed by $g$, a function on $\mathbb Z^2\times D_4$ (the regular representation).
2. *Group convolution:* subsequent layers correlate over the whole group $\mathfrak G = \mathbb Z^2\rtimes D_4$ (translations combined with rotations and flips; called p4m):
   $$ [f\star\psi](g) = \sum_{h\in\mathfrak G}\sum_{k}f_k(h)\,\psi_k(g^{-1}h). $$
3. *Pooling over $G$* at the end yields invariant outputs.

Every layer commutes with the group action, so the whole network is equivariant (see [[Invariant and Equivariant Functions]]).

**Steerable CNNs.** Instead of all 8 copies, features are organised by the **irreducible representations** of the group ($D_4$ has four 1-dimensional and one 2-dimensional irrep). Filters are constrained to an equivariant basis that maps one representation type to another. The kernel constraint $w(gx) = \rho_{\text{out}}(g)\,w(x)\,\rho_{\text{in}}(g)^{-1}$ is solved once; the solutions form a linear subspace. For point offsets, the orbit structure (orbits of size 1, 4 and 8) and the stabiliser subgroups determine which irreps each orbit can carry. This fixes which channels may mix.

**Theory.** For compact groups, every equivariant *linear* layer is necessarily a generalised group convolution (Kondor & Trivedi 2018), so G-CNNs are the natural equivariant architecture. Networks built from such layers and suitable nonlinearities can approximate any continuous equivariant map (Yarotsky 2022). This extends the universality of ordinary deep CNNs (Zhou 2020).

## References

1. T. Cohen, M. Welling (2016). *Group Equivariant Convolutional Networks*. ICML 2016. [arXiv:1602.07576](https://arxiv.org/abs/1602.07576)
2. M. Weiler, G. Cesa (2019). *General E(2)-Equivariant Steerable CNNs*. NeurIPS 32. [arXiv:1911.08251](https://arxiv.org/abs/1911.08251)
3. R. Kondor, S. Trivedi (2018). *On the Generalization of Equivariance and Convolution in Neural Networks to the Action of Compact Groups*. ICML 2018. [arXiv:1802.03690](https://arxiv.org/abs/1802.03690)
4. D. Yarotsky (2022). *Universal Approximations of Invariant Maps by Neural Networks*. Constr. Approx. 55, 407–474. [doi:10.1007/s00365-021-09546-1](https://doi.org/10.1007/s00365-021-09546-1)
5. D.-X. Zhou (2020). *Universality of deep convolutional neural networks*. Appl. Comput. Harmon. Anal. 48(2), 787–794. [doi:10.1016/j.acha.2019.06.004](https://doi.org/10.1016/j.acha.2019.06.004)
