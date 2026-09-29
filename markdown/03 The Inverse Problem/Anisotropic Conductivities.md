---
tags: [inverse-problem, theory]
---

In anisotropic tissue such as muscle, or in layered materials, the conductivity is a symmetric positive definite matrix field $\gamma(x)\in\mathbb R^{n\times n}$, and the equation is $\nabla\cdot(\gamma\nabla u) = 0$.

**Non-uniqueness.** Let $\Phi:\bar\Omega\to\bar\Omega$ be a diffeomorphism with $\Phi|_{\partial\Omega} = \mathrm{id}$. The push-forward

$$
\Phi_*\gamma = \left(\frac{D\Phi\,\gamma\,D\Phi^\top}{|\det D\Phi|}\right)\circ\Phi^{-1}
$$

has the same DtN map, $\Lambda_{\Phi_*\gamma} = \Lambda_\gamma$. This change of variables in the Dirichlet energy was pointed out by Tartar (see Kohn–Vogelius 1984). Anisotropic conductivities can therefore only be recovered **up to such diffeomorphisms**. This is proven in 2D (Sylvester 1990; Astala, Lassas, Päivärinta 2005), and for real-analytic conductivities in $n\ge3$ (Lee–Uhlmann 1989).

**Relevance.**

- The *relaxed* [[Kohn-Vogelius Functional]] produces anisotropic limits, which is one reason relaxation did not lead to a practical algorithm.
- The push-forward formula explains the [[Symmetries of the EIT Problem|symmetries of the forward map]] under rigid motions. In 2D, where conformal maps have $D\Phi D\Phi^\top = |\det D\Phi| I$, isotropic conductivities stay isotropic.

## References

1. R. V. Kohn, M. Vogelius (1984). *Determining conductivity by boundary measurements*. Comm. Pure Appl. Math. 37(3), 289–298. [doi:10.1002/cpa.3160370302](https://doi.org/10.1002/cpa.3160370302)
2. J. M. Lee, G. Uhlmann (1989). *Determining anisotropic real-analytic conductivities by boundary measurements*. Comm. Pure Appl. Math. 42(8), 1097–1112. [doi:10.1002/cpa.3160420804](https://doi.org/10.1002/cpa.3160420804)
3. J. Sylvester (1990). *An anisotropic inverse boundary value problem*. Comm. Pure Appl. Math. 43(2), 201–232. [doi:10.1002/cpa.3160430203](https://doi.org/10.1002/cpa.3160430203)
4. K. Astala, M. Lassas, L. Päivärinta (2005). *Calderón's Inverse Problem for Anisotropic Conductivity in the Plane*. Comm. PDE 30(1–2), 207–224. [doi:10.1081/PDE-200044485](https://doi.org/10.1081/PDE-200044485)
