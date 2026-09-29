---
tags: [machine-learning, theory]
---

The **manifold hypothesis** states that high-dimensional natural data, such as images or anatomical conductivity maps, concentrate near a low-dimensional manifold $\mathcal M$ (or a union of such) inside the ambient space $\mathbb R^n$.

**As a prior for inverse problems.** If $\sigma\in\mathcal M$ with $\dim\mathcal M\ll n$, the effective number of unknowns is small. In finite dimensions EIT becomes Lipschitz stable (see [[Stability of the Calderón Problem]]). A learned representation of $\mathcal M$ can therefore regularise strongly.

**Autoencoder penalty.** Train an encoder $E$ and a decoder $G$ on samples of $\mathcal M$. Then

$$
\mathcal R(\sigma) = \|\sigma-G(E(\sigma))\|^2
$$

measures the distance from the learned manifold, approximately. Its gradient, obtained by automatic differentiation, pushes $\sigma$ towards $\mathcal M$. It is added to the data gradient in the [[Iterative Reconstruction Loop]]:

$$
\sigma_{k+1} = \sigma_k-\tau\big(\nabla\Phi_{\text{data}}(\sigma_k)+\beta\nabla\mathcal R(\sigma_k)\big).
$$

**Alternatives.** Optimise directly over the latent code, $\min_z\Phi_{\text{data}}(G(z))$ (generative-model inversion). This restricts the solution strictly to the range of $G$, which risks model bias. Diffusion models learn the score of a smoothed data distribution, whose mass concentrates near $\mathcal M$ as the noise level goes to zero (see [[Score Function]]).

**Caveats.** Estimating manifolds from samples is statistically and computationally hard in general (Kiani et al. 2024), and real data may only approximately satisfy the hypothesis (Whiteley et al. 2025).

## References

1. L. Cayton (2005). *Algorithms for manifold learning*. Research exam, UC San Diego. [cseweb.ucsd.edu/~lcayton/resexam.pdf](https://cseweb.ucsd.edu/~lcayton/resexam.pdf)
2. N. Whiteley, A. Gray, P. Rubin-Delanchy (2025). *Statistical exploration of the Manifold Hypothesis*. [arXiv:2208.11665](https://arxiv.org/abs/2208.11665)
3. B. T. Kiani, J. Wang, M. Weber (2024). *Hardness of Learning Neural Networks under the Manifold Hypothesis*. [arXiv:2406.01461](https://arxiv.org/abs/2406.01461)
4. J. H. Seidman, G. Kissas, P. Perdikaris, G. J. Pappas (2022). *NOMAD: Nonlinear Manifold Decoders for Operator Learning*. NeurIPS 35. [arXiv:2206.03551](https://arxiv.org/abs/2206.03551)
