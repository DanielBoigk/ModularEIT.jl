---
tags: [state-of-the-art, machine-learning]
---

Deep learning enters EIT reconstruction in several ways:

1. **Post-processing.** A network, typically a [[U-Net]], sharpens an initial reconstruction. Example: *Deep D-bar* (Hamilton & Hauptmann 2018) post-processes [[D-bar Method]] images.
2. **Learned direct inversion.** A network maps measurements straight to images. This is fast but ignores the physics at test time, needs large paired training sets, and generalises poorly beyond them. Tanyu et al. (2023) compare such methods fairly against analytic ones.
3. **Physics-based with learned components.** The forward model stays in the loop, and learning supplies the prior:
   - [[Plug-and-Play Priors]] and [[Regularization by Denoising]] with learned denoisers;
   - *Deep image prior* reconstructions such as DeepEIT (Liu et al. 2023), which parametrise $\sigma$ by an untrained network;
   - learned regularisers (see [[Learned Regularization]]);
   - generative priors from normalising flows, VAEs or score-based models (Wang et al. 2024; see [[Diffusion Models for EIT]]).
4. **Learned solvers and operators.** Neural operators approximate the forward map, or learned iterative schemes unroll an optimiser.

The physics-based family keeps the data consistency of classical methods and uses learning only where knowledge is missing. This makes it the most robust option when training data are scarce.

## References

1. S. J. Hamilton, A. Hauptmann (2018). *Deep D-Bar: Real-Time Electrical Impedance Tomography Imaging With Deep Neural Networks*. IEEE Trans. Med. Imaging 37(10), 2367–2377. [doi:10.1109/TMI.2018.2828303](https://doi.org/10.1109/TMI.2018.2828303)
2. D. N. Tanyu et al. (2023). *Deep learning methods for partial differential equations and related parameter identification problems*. Inverse Problems 39(10), 103001. [doi:10.1088/1361-6420/ace9d4](https://doi.org/10.1088/1361-6420/ace9d4)
3. D. Liu, J. Wang, Q. Shan, D. Smyl, J. Deng, J. Du (2023). *DeepEIT: Deep Image Prior Enabled Electrical Impedance Tomography*. IEEE TPAMI 45(8), 9627–9638. [doi:10.1109/TPAMI.2023.3240565](https://doi.org/10.1109/TPAMI.2023.3240565)
4. H. Wang, G. Xu, Q. Zhou (2024). *A Comparative Study of Variational Autoencoders, Normalizing Flows, and Score-based Diffusion Models for Electrical Impedance Tomography*. [arXiv:2310.15831](https://arxiv.org/abs/2310.15831)
5. S. Arridge, P. Maass, O. Öktem, C.-B. Schönlieb (2019). *Solving inverse problems using data-driven models*. Acta Numerica 28, 1–174. [doi:10.1017/S0962492919000059](https://doi.org/10.1017/S0962492919000059)
