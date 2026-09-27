---
tags: [geometric-learning, forward-problem]
---

The EIT [[Forward Map]] commutes with several transformations. This is useful both for analysis and for designing neural networks that respect the physics (see [[Invariant and Equivariant Functions]]).

**Rigid motions (Euclidean group $E(n)$).** Let $T(x) = Qx+b$ with $Q\in O(n)$. If $u$ solves $\nabla\cdot(\gamma\nabla u) = 0$ in $\Omega$, then $u\circ T^{-1}$ solves the equation with conductivity $\gamma\circ T^{-1}$ in $T(\Omega)$. The normal derivative is preserved because $T$ is an isometry, so

$$
\Lambda_{\gamma\circ T^{-1}}\big(f\circ T^{-1}\big) = \big(\Lambda_\gamma f\big)\circ T^{-1}.
$$

Rotating or reflecting the body and the boundary data together rotates or reflects the measured currents. The forward map is *equivariant*, not invariant: an interior feature is reconstructed the same way wherever it sits, provided the whole setup (domain and electrodes) is transformed with it. On a square domain the relevant finite subgroup is the [[Dihedral Group D4]].

**Scaling.** For $x\mapsto cx$ ($c>0$), gradients scale by $1/c$, so the DtN map of the dilated problem is $\frac1c$ times the transported one. Multiplying the conductivity by a constant gives $\Lambda_{c\gamma} = c\Lambda_\gamma$.

**Conformal maps (2D).** For a conformal map $\Phi$, $D\Phi\,D\Phi^\top = |\det D\Phi|\,I$. The push-forward of an isotropic conductivity is therefore again isotropic, $\Phi_*\gamma = \gamma\circ\Phi^{-1}$ (see [[Anisotropic Conductivities]]). The 2D conductivity equation is conformally invariant. The boundary operators transform by composition with $\Phi$ and multiplication by the boundary stretching factor $|\Phi'|$. This much larger (infinite-dimensional) symmetry is special to two dimensions and is used, for example, to map arbitrary simply connected domains to the unit disc.

**Mesh permutations.** Renumbering the nodes of a finite element mesh (the symmetric group $S_N$) changes nothing physically. Graph neural networks are permutation-equivariant by construction.

**Complex EIT.** A global phase rotation $e^{i\phi}$ ($U(1)$) of the complex potentials leaves the admittivity problem unchanged (see [[Complex Conductivity]]).

## References

1. R. V. Kohn, M. Vogelius (1984). *Determining conductivity by boundary measurements*. Comm. Pure Appl. Math. 37(3), 289–298. [doi:10.1002/cpa.3160370302](https://doi.org/10.1002/cpa.3160370302)
2. J. Sylvester (1990). *An anisotropic inverse boundary value problem*. Comm. Pure Appl. Math. 43(2), 201–232. [doi:10.1002/cpa.3160430203](https://doi.org/10.1002/cpa.3160430203)
3. M. M. Bronstein, J. Bruna, T. Cohen, P. Veličković (2021). *Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges*. [arXiv:2104.13478](https://arxiv.org/abs/2104.13478)
