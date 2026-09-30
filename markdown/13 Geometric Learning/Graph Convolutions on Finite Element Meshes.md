---
tags: [machine-learning, geometry, architecture, fem]
aliases: [Mesh convolutions, Graph neural networks on meshes, GCN on FEM meshes, Message passing on meshes]
---

A convolution needs three things: a notion of neighbourhood, weights that do not depend on the position, and a way to aggregate. On a pixel grid all three come from the $n\times n$ stencil. On a finite element mesh the neighbourhood is the mesh graph (nodes sharing an element), and there are three ways to build the weights.

## Message passing

Every node $i$ with features $h_i$ and position $x_i$ collects messages from its neighbours $j$:

$$
h_i' = \phi\Big(h_i,\ \sum_{j\in\mathcal N(i)}\psi\big(h_i,\ h_j,\ x_j - x_i\big)\Big),
$$

with small networks $\phi,\psi$ shared by all nodes. Without the relative position $x_j-x_i$ the filter is **isotropic**: it cannot tell up from down or detect edges of a given orientation. With it, the network can learn directional filters on any mesh, like a stencil. MeshGraphNets (Pfaff et al. 2021) use this form for simulations on unstructured meshes.

## Polynomial filters of the Laplacian

A graph convolution in the spectral sense is a function of a Laplacian $L$, evaluated as a polynomial so that it stays local:

$$
y = \sum_{k=0}^{K}\theta_k\,T_k(\hat L)\,h ,
$$

with Chebyshev polynomials $T_k$ of the rescaled operator $\hat L$ (ChebNet, Defferrard et al. 2016; GCN is the case $K = 1$, Kipf and Welling 2017). Degree $K$ reaches $K$ edges, like a stencil of radius $K$. The combinatorial graph Laplacian depends on how the mesh is drawn. On a finite element mesh, however, the natural choice is the discrete Laplace operator

$$
L = M^{-1}K,
$$

with the [[Mass Matrix]] $M$ (lumped) and the [[Stiffness Matrix]] $K$. Its low eigenvectors approximate those of the continuous Laplacian, so the filters mean the same thing on every mesh. Refining the mesh changes the discretisation but not the filter. The price is isotropy: polynomials of the Laplacian are rotation invariant.

## Continuous kernels

The convolution integral $y(x) = \int\kappa_\theta(x'-x)\,h(x')\,\mathrm dx'$, with a learned kernel $\kappa_\theta$ (a small network or B-spline, SplineCNN, Fey et al. 2018), is evaluated by quadrature over the neighbours:

$$
y_i = \sum_{j\in\mathcal N(i)} w_j\,\kappa_\theta(x_j - x_i)\,h_j ,
$$

with weights $w_j$ from the lumped mass matrix. The kernel is a function of the continuous offset, anisotropic if it needs to be, and consistent under refinement. Graph kernel networks, a form of neural operator (Li et al. 2020), use exactly this.

## Pooling: the mesh hierarchy

A U-Net needs coarser levels. On a mesh they come for free from mesh refinement: a hierarchy produced by [[Newest Vertex Bisection]] or quadtree refinement (see [[Hanging Nodes]], [[Coarsening of Bisection Meshes]]) has a coarse mesh on every level, with the prolongation of finite element interpolation between them. The down path restricts (the transpose of the prolongation, mass-weighted), and the up path prolongates. This is the structure of a multigrid V-cycle, with learned smoothers. Graph U-Nets (Gao and Ji 2019) learn the pooling instead, which works on any graph but loses the geometric meaning of the levels.

## For EIT

- **Resolution where the data are.** A mesh refined towards the boundary (see [[Adaptive Meshing in EIT]]) gives the network fine resolution exactly where the measurements determine the conductivity, which pixels cannot.
- **Model-based learning.** Herzberg et al. (2021) use graph convolutions on the finite element mesh of the forward model inside a learned Gauss–Newton iteration for EIT, so network and physics share the discretisation, and the method works on different domain shapes.
- **Cost.** Sparse gather/scatter operations are slower per node than dense convolutions, and meshes of different sizes must be batched as disjoint graphs. For a fixed domain with modest resolution, pixels with a parametrisation (see [[Parametrizations of the Conductivity]]) are simpler and faster.
- **Noise and priors.** Diffusion models on mesh functions need noise that is discretisation-consistent, i.e. weighted with the mass matrix (see [[Diffusion Models on Finite Element Spaces]]).

## References

1. T. N. Kipf, M. Welling (2017). *Semi-Supervised Classification with Graph Convolutional Networks*. ICLR 2017. [arXiv:1609.02907](https://arxiv.org/abs/1609.02907)
2. M. Defferrard, X. Bresson, P. Vandergheynst (2016). *Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering*. NeurIPS 2016. [arXiv:1606.09375](https://arxiv.org/abs/1606.09375)
3. T. Pfaff, M. Fortunato, A. Sanchez-Gonzalez, P. W. Battaglia (2021). *Learning Mesh-Based Simulation with Graph Networks*. ICLR 2021. [arXiv:2010.03409](https://arxiv.org/abs/2010.03409)
4. M. Fey, J. E. Lenssen, F. Weichert, H. Müller (2018). *SplineCNN: Fast Geometric Deep Learning with Continuous B-Spline Kernels*. CVPR 2018. [doi:10.1109/CVPR.2018.00097](https://doi.org/10.1109/CVPR.2018.00097)
5. Z. Li, N. Kovachki, K. Azizzadenesheli, B. Liu, K. Bhattacharya, A. Stuart, A. Anandkumar (2020). *Neural Operator: Graph Kernel Network for Partial Differential Equations*. [arXiv:2003.03485](https://arxiv.org/abs/2003.03485)
6. H. Gao, S. Ji (2019). *Graph U-Nets*. ICML 2019. [arXiv:1905.05178](https://arxiv.org/abs/1905.05178)
7. W. Herzberg, D. B. Rowe, A. Hauptmann, S. J. Hamilton (2021). *Graph Convolutional Networks for Model-Based Learning in Nonlinear Inverse Problems*. IEEE Trans. Comput. Imaging 7, 1341–1353. [doi:10.1109/TCI.2021.3132190](https://doi.org/10.1109/TCI.2021.3132190)
