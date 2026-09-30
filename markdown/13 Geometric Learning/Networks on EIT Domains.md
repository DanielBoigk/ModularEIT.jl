---
tags: [machine-learning, geometry, architecture]
aliases: [Network architectures for EIT, Domain-agnostic networks, Networks on non-rectangular domains]
---

Convolutional networks are built for pixel images: a rectangular grid, uniform resolution, and translation-invariant $n\times n$ stencils. EIT conductivities live on domains of any shape (a disk, a thorax, a head), are discretised on unstructured and often adaptively refined meshes (see [[Adaptive Meshing in EIT]]), and are determined by the data very unevenly: sharply near the boundary, hardly at all in the interior (see [[Resolution and Confidence Maps]]). A network that is to act as a prior or a denoiser for EIT has to bridge this gap. There are four ways to do it.

| approach | domain shape | resolution | cost | main difficulty |
|---|---|---|---|---|
| pixels on a bounding box | fixed | uniform | dense convolutions, fast | the boundary, where the data are, is staircased and implicit |
| [[Masked and Partial Convolutions\|masked or partial convolutions]] | any, given as a mask | uniform | dense, fast | still a uniform grid; the network must learn every boundary shape it meets |
| [[Conformal Transplantation of Networks\|conformal transplantation]] | any simply connected 2D domain | graded by the map | dense on the reference domain | the image statistics are distorted by the map |
| [[Graph Convolutions on Finite Element Meshes\|convolutions on the mesh]] | any | the mesh's own, e.g. refined at the boundary | sparse, slower per node | isotropic filters, pooling, batching |

## Pixels on a bounding box

The simplest option rasterises the conductivity onto a rectangle containing the domain and sets the outside to a constant (see [[Pixel Images and Finite Element Functions]]). Most published deep-learning EIT methods do this for a fixed disk: the network sees the same outline in every training image and learns to ignore the outside. It breaks down as soon as the shape varies. Its resolution is uniform, which is too coarse at the boundary, where the data are informative, and too fine in the interior, where they are not.

## What makes EIT special

- **The boundary matters most.** The measurements determine the conductivity best near the electrodes. Architectures that treat the boundary as an afterthought (zero padding, staircasing) lose resolution exactly where it exists.
- **Orientation may matter.** Physical EIT is invariant under the symmetries of the domain and the electrode array (see [[Symmetries of the EIT Problem]]). Image priors, however, may have a preferred orientation, as natural scenes do: sky above ground. Isotropic mesh convolutions must then be given the coordinates as features.
- **Discretisation consistency.** The same conductivity on two meshes should get the same prior. Noise and norms must therefore be the discretisations of function-space objects, with mass matrices, not per-node white noise (see [[Diffusion Models on Finite Element Spaces]]).

## A practical combination

The unknowns can be decoupled from the mesh: a network works on pixels, the forward model on an adapted mesh, with a [[Parametrizations of the Conductivity|pixel parametrisation]] in between. With a confidence-weighted data-consistency step (see [[DiffPIR]]), the data determine what they can, near the boundary, and the network fills in the rest. That is the approach of EITDenoiser.jl. Native mesh networks are the alternative when the domain varies strongly or the resolution must follow the mesh.

## References

1. M. M. Bronstein, J. Bruna, T. Cohen, P. Veličković (2021). *Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges*. [arXiv:2104.13478](https://arxiv.org/abs/2104.13478)
2. S. J. Hamilton, A. Hauptmann (2018). *Deep D-Bar: Real-Time Electrical Impedance Tomography Imaging With Deep Neural Networks*. IEEE Trans. Med. Imaging 37(10), 2367–2377. [doi:10.1109/TMI.2018.2828303](https://doi.org/10.1109/TMI.2018.2828303)
3. W. Herzberg, D. B. Rowe, A. Hauptmann, S. J. Hamilton (2021). *Graph Convolutional Networks for Model-Based Learning in Nonlinear Inverse Problems*. IEEE Trans. Comput. Imaging 7, 1341–1353. [doi:10.1109/TCI.2021.3132190](https://doi.org/10.1109/TCI.2021.3132190)
