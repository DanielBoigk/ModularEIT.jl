---
tags: [machine-learning, architecture]
aliases: [ICNN]
---

An **input convex neural network (ICNN)** is a network $f_\theta(x)$ that is guaranteed to be convex in its input $x$. For layers

$$
z_{k+1} = \phi\big(W^{(z)}_kz_k+W^{(x)}_kx+b_k\big),\qquad f_\theta(x) = z_K,
$$

convexity holds if all $W^{(z)}_k$ have **non-negative** entries and the activation $\phi$ is convex and non-decreasing (e.g. ReLU, softplus). The *passthrough* weights $W^{(x)}_k$ are unconstrained.

**Why useful for inverse problems.**

- A convex learned regulariser $\mathcal R_\theta = f_\theta$ gives a convex variational problem when the data term is convex. Minimisers are then global, prox operators are well defined, and stability and convergence results carry over (Mukherjee et al.; Goujon et al.).
- Gradients of ICNNs are monotone maps. In optimal transport, the Brenier map is the gradient of a convex potential, so ICNNs parametrise transport maps and can define learned, convex **metrics or divergences**, for example as [[Data Fidelity Terms]].

EIT's data term is nonconvex, so convexity of the regulariser does not make the whole problem convex. It does make the regulariser step in [[ADMM]] well posed and stable.

## References

1. B. Amos, L. Xu, J. Z. Kolter (2017). *Input Convex Neural Networks*. ICML 2017. [arXiv:1609.07152](https://arxiv.org/abs/1609.07152)
2. S. Mukherjee, S. Dittmer, Z. Shumaylov, S. Lunz, O. Öktem, C.-B. Schönlieb (2021). *Learned convex regularizers for inverse problems*. [arXiv:2008.02839](https://arxiv.org/abs/2008.02839)
3. A. Goujon, S. Neumayer, P. Bohra, S. Ducotterd, M. Unser (2023). *A Neural-Network-Based Convex Regularizer for Inverse Problems*. [arXiv:2211.12461](https://arxiv.org/abs/2211.12461)
