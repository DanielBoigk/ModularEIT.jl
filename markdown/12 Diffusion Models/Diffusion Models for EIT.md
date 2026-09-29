---
tags: [state-of-the-art, diffusion]
---

Recent work that combines diffusion or score-based priors with EIT:

- **Diff-INR** (Tong, Wang, Liu 2024): represents the conductivity as an implicit neural representation and uses a pretrained diffusion model as a generative regulariser during the reconstruction.
- **Conditional diffusion** (Shi, Kang, Liatsis 2024): trains a diffusion model conditioned on the boundary measurements, so boundary-operator information enters the generative model directly.
- **Diffusion graph posterior sampling** (Alberti, Lazzaro, Morigi, Santacesaria, Wang 2026): adapts the diffusion architecture to conductivities on unstructured FEM meshes via graph networks, instead of pixel grids, for nonlinear inverse problems such as EIT.
- **Comparative study** (Wang, Xu, Zhou 2024): VAEs, normalising flows and score-based diffusion models as priors for EIT, within a common Bayesian framework.

**Recurring themes.**

- EIT's forward operator is expensive and nonlinear. Methods that need few forward solves or tolerate inexact data steps ([[DiffPIR]], [[RED-Diff]]) are more practical than those that backpropagate through the network at every step ([[Diffusion Posterior Sampling]]).
- Pixel-grid diffusion models do not match FEM meshes or curved domains. Mask channels, graph networks or implicit representations bridge the gap (see [[U-Net]]).
- Weakly determined interiors invite [[Hallucinations and Uncertainty|hallucinations]]. Multiple posterior samples are the natural uncertainty estimate.

## References

1. B. Tong, J. Wang, D. Liu (2024). *Diff-INR: Generative Regularization for Electrical Impedance Tomography*. [arXiv:2409.04494](https://arxiv.org/abs/2409.04494)
2. S. Shi, R. Kang, P. Liatsis (2024). *A Conditional Diffusion Model for Electrical Impedance Tomography Image Reconstruction*. [arXiv:2412.16979](https://arxiv.org/abs/2412.16979)
3. G. S. Alberti, D. Lazzaro, S. Morigi, M. Santacesaria, S. Wang (2026). *Diffusion Graph Posterior Sampling for Nonlinear Inverse Problems with Application to Electrical Impedance Tomography*. [arXiv:2605.19621](https://arxiv.org/abs/2605.19621)
4. H. Wang, G. Xu, Q. Zhou (2024). *A Comparative Study of Variational Autoencoders, Normalizing Flows, and Score-based Diffusion Models for Electrical Impedance Tomography*. [arXiv:2310.15831](https://arxiv.org/abs/2310.15831)
