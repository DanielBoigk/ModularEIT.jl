---
tags: [machine-learning, architecture]
---

The **U-Net** is a convolutional encoder–decoder with skip connections, the standard backbone for image-to-image tasks and for diffusion-model denoisers $\varepsilon_\theta(x,t)$.

- **Encoder:** convolution blocks with downsampling, halving the resolution and increasing the channels. This builds a multi-scale representation with a growing receptive field.
- **Bottleneck:** coarsest resolution. Modern variants place self-attention or transformer blocks here to capture global structure.
- **Decoder:** upsampling blocks that restore the resolution.
- **Skip connections:** concatenate encoder features at each scale into the decoder, preserving fine detail lost in downsampling.

For diffusion models, the noise level or time $t$ enters through a [[Sinusoidal Time Embedding]] added to every block, and residual blocks with group normalisation are used.

**Local vs. global models.** A small CNN with a limited receptive field learns local texture statistics but not global composition. A U-Net sees the whole image and learns global structure with a similar parameter count. Plain convolutions assume a regular grid and do not "see" non-rectangular domains; an extra *mask channel* (1 inside the domain, 0 outside) lets the network learn the boundary (see [[Masked and Partial Convolutions]]). For unstructured FEM meshes, graph neural networks are the analogue (see [[Graph Convolutions on Finite Element Meshes]], and [[Networks on EIT Domains]] for the options).

## References

1. O. Ronneberger, P. Fischer, T. Brox (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation*. MICCAI 2015. [arXiv:1505.04597](https://arxiv.org/abs/1505.04597)
2. J. Ho, A. Jain, P. Abbeel (2020). *Denoising Diffusion Probabilistic Models*. NeurIPS 33. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
