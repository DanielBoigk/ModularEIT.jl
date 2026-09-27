---
tags: [machine-learning, architecture]
aliases: [DEQ, Implicit networks]
---

A **deep equilibrium model (DEQ)** defines its output as a fixed point of a single layer:

$$
z^* = f_\theta(z^*,x).
$$

Formally this corresponds to an infinitely deep, weight-tied network. The forward pass finds $z^*$ with a root-finder (fixed-point iteration, Anderson acceleration, Broyden).

**Implicit differentiation.** Gradients do not require backpropagating through the solver iterations. By the implicit function theorem,

$$
\frac{\partial z^*}{\partial\theta} = \big(I-\partial_zf_\theta(z^*,x)\big)^{-1}\partial_\theta f_\theta(z^*,x),
$$

and the vector–Jacobian product needs one linear solve with $(I-\partial_zf_\theta)^\top$. That is again an adjoint equation.

**Existence and stability.** A unique fixed point that plain iteration finds requires, for example, that $f_\theta(\cdot,x)$ be a **contraction** (Banach fixed-point theorem). Unconstrained architectures do not guarantee this, and training can become unstable. Constructions with guarantees include monotone operator equilibrium networks, spectral normalisation, and parametrisations that bound $\|\partial_zf_\theta\|<1$.

**In inverse problems.** DEQs can represent the *converged* state of a learned iterative reconstruction. An example is a fixed point of "data step + learned denoiser", as in [[Plug-and-Play Priors]] and RED (Gilton et al. 2021; Liu et al. 2022).

## References

1. S. Bai, J. Z. Kolter, V. Koltun (2019). *Deep Equilibrium Models*. NeurIPS 32. [arXiv:1909.01377](https://arxiv.org/abs/1909.01377)
2. E. Winston, J. Z. Kolter (2020). *Monotone operator equilibrium networks*. NeurIPS 33. [arXiv:2006.08591](https://arxiv.org/abs/2006.08591)
3. D. Gilton, G. Ongie, R. Willett (2021). *Deep Equilibrium Architectures for Inverse Problems in Imaging*. IEEE Trans. Comput. Imaging 7. [arXiv:2102.07944](https://arxiv.org/abs/2102.07944)
4. J. Liu, X. Xu, W. Gan, S. Shoushtari, U. S. Kamilov (2022). *Online Deep Equilibrium Learning for Regularization by Denoising*. NeurIPS 35. [proceedings.neurips.cc](https://proceedings.neurips.cc/paper_files/paper/2022/hash/a2440e23f6a8c037eff1dc4f1156aa35-Abstract-Conference.html)
5. A. Pal, A. Edelman, C. Rackauckas (2023). *Continuous Deep Equilibrium Models: Training Neural ODEs faster by integrating them to Infinity*. [arXiv:2201.12240](https://arxiv.org/abs/2201.12240)
