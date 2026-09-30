---
tags: [machine-learning, geometry, architecture]
aliases: [Masked convolution, Partial convolution, Normalized convolution, Mask channel]
---

A domain $\Omega$ inside a pixel grid is described by a **mask** $m_{ij}\in\{0,1\}$: 1 for pixels inside, 0 outside (see [[Pixel Images and Finite Element Functions]]). A convolution can use it in two ways.

## Mask as an input channel

The mask is concatenated to the features before every convolution:

$$
y = W * [x,\ m] + b .
$$

The network then *sees* where the boundary is at every depth. Because the mask is zero-padded like everything else, pixels near the boundary of the domain (and of the image) see a distinctive pattern of mask values. The network can learn to treat them differently. Nothing forces it to ignore the values outside. It has to learn that from the training data, which must therefore contain the masks it will meet.

Pooling must keep the mask aligned. A coarse pixel counts as inside if any of its fine pixels is (max-pooling of the mask). Upsampling copies it.

## Partial convolutions

A partial convolution uses only the valid pixels of every window and rescales by their fraction (Liu et al. 2018; normalised convolution, Knutsson and Westin 1993):

$$
y_{ij} = \begin{cases}\displaystyle W^\top(x\odot m)_{\mathcal N(ij)}\;\frac{|\mathcal N|}{\sum m_{\mathcal N(ij)}} + b, & \sum m_{\mathcal N(ij)}>0,\\[4pt] 0, & \text{otherwise},\end{cases}
\qquad m'_{ij} = \mathbb 1\big[\textstyle\sum m_{\mathcal N(ij)}>0\big].
$$

The output is then independent of the values outside the domain *by construction*. Near the boundary a window is simply averaged over fewer pixels. The updated mask $m'$ grows by one stencil radius per layer. That was designed for inpainting, where holes should be filled; on a fixed domain one keeps the original mask instead.

## Does it work for non-rectangular domains?

Yes. Both variants turn any domain on a pixel grid into something a standard U-Net can process, and the domain may change from sample to sample. The remaining limitations come from the grid, not the mask:

- The boundary is **staircased**. Its geometry enters only at pixel resolution, while EIT data are most informative right at the boundary.
- The resolution is **uniform**. The data resolve the boundary layer finely and the interior coarsely (see [[Resolution and Confidence Maps]]), so a uniform grid is too coarse in one place and wastes parameters in the other.
- **Generalisation across shapes must be trained.** A network trained on one outline has no reason to handle another. Training with random masks (random domains, or random crops of images) makes it shape-agnostic within the family of shapes it has seen.

## Why it is not used more

Most learned EIT reconstructions are trained and tested on one fixed geometry, usually a disk. For a fixed domain the mask is a constant input and adds nothing that a network cannot learn implicitly. Masked and partial convolutions pay off when the domain varies (patient-specific thorax outlines, different tanks), and such data sets are rare. Where shapes vary, conformal maps to a reference domain (see [[Conformal Transplantation of Networks]]) or networks on the mesh itself (see [[Graph Convolutions on Finite Element Meshes]]) handle the boundary more faithfully than a staircased mask.

A related use is the *local* score network: only small stencils, no pooling, and no normalisation over space, so that each output depends only on a neighbourhood. The mask channel then tells every pixel how close it is to the boundary. Such a network models texture, not layout, and suits a prior for small-scale structure that is combined with a data term for the large scales.

**In EITDenoiser.jl:** the local score network with mask-channel convolutions (`LocalScoreNet`, `MaskedConv`), see the [repository](https://github.com/DanielBoigk/EITDenoiser.jl).

## References

1. G. Liu, F. A. Reda, K. J. Shih, T.-C. Wang, A. Tao, B. Catanzaro (2018). *Image Inpainting for Irregular Holes Using Partial Convolutions*. ECCV 2018. [doi:10.1007/978-3-030-01252-6_6](https://doi.org/10.1007/978-3-030-01252-6_6)
2. H. Knutsson, C.-F. Westin (1993). *Normalized and differential convolution*. CVPR 1993. [doi:10.1109/CVPR.1993.341081](https://doi.org/10.1109/CVPR.1993.341081)
