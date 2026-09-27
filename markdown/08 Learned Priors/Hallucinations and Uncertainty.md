---
tags: [machine-learning, reliability]
aliases: [Hallucination]
---

A **hallucination** is image content produced by the prior rather than supported by the data: a plausible-looking feature that is not there, or a missing feature that is.

**Why EIT is especially exposed.** Boundary data determine the interior only weakly (see [[Stability of the Calderón Problem]] and [[Decay of Boundary Measurements]]). Many very different interiors fit the data equally well within the noise, and a strong generative prior will "fill in" the interior with typical training content.

**Mitigations.**

- **Data consistency checks:** the final image must reproduce the measured data to within the noise (discrepancy principle). Hard projection or [[Proximal Operator|prox]] steps onto the data-consistent set are stricter than soft penalties.
- **Physics-aware priors:** constrain the prior's updates to directions the data cannot see, or penalise changes that alter the predicted currents. One option is an operator-weighted metric in the prox, $\|x-x_0\|_{W}$ with $W$ built from the forward operator.
- **Uncertainty quantification:** draw multiple posterior samples (different noise seeds in [[Diffusion Posterior Sampling]] or [[DiffPIR]]) and report the pixel-wise mean and variance. High-variance regions are where the prior dominates. Calibrated methods, such as conformal prediction, give coverage guarantees.
- **Prior mismatch testing:** evaluate on out-of-distribution conductivities.

## References

1. S. Bhadra, V. A. Kelkar, F. J. Brooks, M. A. Anastasio (2021). *On Hallucinations in Tomographic Image Reconstruction*. IEEE Trans. Med. Imaging 40(11), 3249–3260. [doi:10.1109/TMI.2021.3077857](https://doi.org/10.1109/TMI.2021.3077857)
2. V. Antun, F. Renna, C. Poon, B. Adcock, A. C. Hansen (2020). *On instabilities of deep learning in image reconstruction and the potential costs of AI*. PNAS 117(48), 30088–30095. [doi:10.1073/pnas.1907377117](https://doi.org/10.1073/pnas.1907377117)
