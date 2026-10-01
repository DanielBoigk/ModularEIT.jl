---
tags: [numerics, fem, adaptivity]
aliases: [Non-conforming meshes, Conformity constraints]
---

Refining a quadrilateral into four children places a new vertex at the midpoint of each edge. If the neighbour across that edge is not refined, the new vertex is a **hanging node**: it is a vertex of the fine cells but lies in the interior of an edge of the coarse cell.

```tikz
\begin{document}
\begin{tikzpicture}[scale=1.6]
  \draw[thick] (0,0) rectangle (2,2);
  \draw[thick] (2,0) rectangle (4,2);
  \draw (0,1) -- (2,1);
  \draw (1,0) -- (1,2);
  \fill (2,1) circle (0.06);
  \node[right] at (2.05,1.12) {hanging node $h$};
  \fill (2,0) circle (0.04);
  \fill (2,2) circle (0.04);
  \node[below] at (2,0) {$m_1$};
  \node[above] at (2,2) {$m_2$};
\end{tikzpicture}
\end{document}
```

**Conformity.** A continuous $Q_1$ function on the fine cells is linear along the coarse edge from the coarse side. So its value at the hanging node is fixed by the two edge end points (the *masters*):

$$
u_h = \tfrac12\,(u_{m_1}+u_{m_2}).
$$

In 3D a hanging node in the middle of a face is the mean of the four face corners. With these constraints, the finite element space is conforming, i.e. a subspace of $H^1(\Omega)$, and all standard error estimates hold.

**Condensation.** Collect the constraints as $\mathbf u_{\mathrm{all}} = C\,\mathbf u$, where $C$ is the identity on free nodes and holds the weights $\tfrac12$ in the rows of hanging nodes. The Galerkin method on the constrained space uses

$$
A_C = C^\top A\,C,\qquad \mathbf f_C = C^\top\mathbf f,
$$

and measurements $Q\,C$. $A_C$ is again symmetric positive semidefinite. Since $C\mathbf 1 = \mathbf 1$, constants remain its null space, and the grounding of the Neumann problem is unchanged (see [[Grounding of the Potential]]). For $A(\sigma) = \sum_a\sigma_a A_a$, condensation commutes with the parameter dependence, so the [[Conductivity Tensor]] condenses once per mesh.

**2:1 balance.** Neighbouring cells may differ by at most one refinement level. Then every hanging node has conforming masters, and the constraints do not chain. Tree-based mesh libraries (forests of quadtrees/octrees) enforce the balance after each refinement.

**Boundary.** In 2D, a hanging node always lies on an edge shared by two cells, never on the domain boundary. Boundary data and electrodes therefore only involve free nodes.

**Piecewise constant conductivities** need no constraints: they are discontinuous anyway. A coarse cell simply has two fine neighbours across the refined edge. For jump-based functionals such as the [[Total Variation]], each fine facet is paired with the coarse cell.

**In ModularEIT.jl:** [`AdaptiveMesh`](https://danielboigk.github.io/ModularEIT.jl/dev/api/adaptivity/#ModularEITFerrite.AdaptiveMesh), [`is_nonconforming`](https://danielboigk.github.io/ModularEIT.jl/dev/api/adaptivity/#ModularEITFerrite.is_nonconforming).

## References

1. P. Šolín, J. Červený, I. Doležel (2008). *Arbitrary-level hanging nodes and automatic adaptivity in the hp-FEM*. Math. Comput. Simul. 77(1), 117–132. [doi:10.1016/j.matcom.2007.02.011](https://doi.org/10.1016/j.matcom.2007.02.011)
2. C. Burstedde, L. C. Wilcox, O. Ghattas (2011). *p4est: Scalable Algorithms for Parallel Adaptive Mesh Refinement on Forests of Octrees*. SIAM J. Sci. Comput. 33(3), 1103–1133. [doi:10.1137/100791634](https://doi.org/10.1137/100791634)
3. M. Ainsworth, J. T. Oden (2000). *A Posteriori Error Estimation in Finite Element Analysis*. Wiley. [doi:10.1002/9781118032824](https://doi.org/10.1002/9781118032824)
