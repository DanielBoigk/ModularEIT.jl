---
tags: [geometric-learning]
aliases: [Group averaging, Symmetrization]
---

The **Reynolds operator** turns an arbitrary function into an invariant one by averaging over a group $G$ with Haar measure $\mu$:

$$
(\mathcal P_Gf)(x) = \frac1{\mu(G)}\int_Gf\big(\rho_X(g)\,x\big)\,\mathrm dg .
$$

The **equivariant** version also transforms the output back:

$$
(\mathcal P^{\text{eq}}_Gf)(x) = \frac1{\mu(G)}\int_G\rho_Y(g)^{-1}f\big(\rho_X(g)\,x\big)\,\mathrm dg .
$$

For a finite group, the integral becomes a sum. For the [[Dihedral Group D4]]:

$$
(\mathcal P_{D_4}f)(x) = \frac18\sum_{g\in D_4}f\big(\rho_X(g)\,x\big).
$$

**Properties.**

- $\mathcal P_Gf$ is invariant (respectively equivariant), and $\mathcal P_Gf = f$ if $f$ already is. So $\mathcal P_G$ is a projection.
- For unitary representations, $\mathcal P_G$ is the *orthogonal* projection in $L^2$ onto the subspace of invariant (equivariant) functions. It gives the best invariant approximation of $f$ in the $L^2$ sense.

**Uses.** Symmetrising convolution filters to obtain invariant filters (see [[Invariant Filter Banks]]). Test-time augmentation, which averages a network's predictions over transformed inputs. Symmetrising learned priors or denoisers so that they respect the [[Symmetries of the EIT Problem]]. The cost grows with $|G|$, while constrained architectures (see [[Equivariant Convolutions]]) have equivariance built in at no extra runtime cost.

## References

1. M. Reisert, H. Burkhardt (2007). *Learning Equivariant Functions with Matrix Valued Kernels*. J. Mach. Learn. Res. 8, 385–408. [jmlr.org/papers/v8/reisert07a.html](https://jmlr.org/papers/v8/reisert07a.html)
2. B. Sturmfels (2008). *Algorithms in Invariant Theory*, 2nd ed. Springer. [doi:10.1007/978-3-211-77417-5](https://doi.org/10.1007/978-3-211-77417-5)
3. M. Finzi, M. Welling, A. G. Wilson (2021). *A Practical Method for Constructing Equivariant Multilayer Perceptrons for Arbitrary Matrix Groups*. ICML 2021. [arXiv:2104.09459](https://arxiv.org/abs/2104.09459)
