---
tags: [numerics, fem, adaptivity]
aliases: [NVB, Bisection refinement]
---

**Newest vertex bisection (NVB)** refines triangular meshes without creating [[Hanging Nodes]]. Every triangle has one designated *refinement edge*, the edge opposite its newest vertex. Refining a triangle bisects this edge: the midpoint becomes the newest vertex of both children, and their refinement edges are the edges opposite it.

**Conformity by closure.** A bisected edge is shared with a neighbour. If the neighbour's refinement edge is the same edge, both are bisected together. Otherwise the neighbour is bisected first along its own refinement edge, possibly recursively, until the shared edge can be bisected. This closure keeps the mesh conforming. The number of extra refinements it causes is bounded by a constant times the number of marked cells.

**Shape regularity.** All triangles produced from an initial triangle fall into finitely many similarity classes (at most four for NVB). The minimum angle stays bounded away from zero, however often the mesh is refined. This makes NVB the standard refinement for the convergence and optimality theory of adaptive finite element methods (see [[Dörfler Marking]]).

**Relatives.** Longest-edge bisection (Rivara) always bisects the longest edge. Red-green refinement splits marked triangles into four congruent children (red) and closes the mesh with temporary bisections (green). Coarsening reverses bisections of sibling pairs (see [[Coarsening of Bisection Meshes]]).

**Comparison with quadtrees.** Quadtree refinement of quadrilaterals keeps a Cartesian structure, well suited to pixel images. It needs hanging-node constraints and a 2:1 balance. Bisection of triangles gives conforming meshes and follows curved boundaries and electrodes better (see [[Adaptive Meshing in EIT]]).

**In ModularEIT.jl:** [`AdaptiveMesh`](https://danielboigk.github.io/ModularEIT.jl/dev/api/adaptivity/#ModularEIT.AdaptiveMesh), [`refine_mesh!`](https://danielboigk.github.io/ModularEIT.jl/dev/api/adaptivity/#ModularEIT.refine_mesh!).

## References

1. W. F. Mitchell (1991). *Adaptive refinement for arbitrary finite-element spaces with hierarchical bases*. J. Comput. Appl. Math. 36(1), 65–78. [doi:10.1016/0377-0427(91)90226-A](https://doi.org/10.1016/0377-0427(91)90226-A)
2. P. Binev, W. Dahmen, R. DeVore (2004). *Adaptive Finite Element Methods with convergence rates*. Numer. Math. 97(2), 219–268. [doi:10.1007/s00211-003-0492-7](https://doi.org/10.1007/s00211-003-0492-7)
3. M.-C. Rivara (1984). *Algorithms for refining triangular grids suitable for adaptive and multigrid techniques*. Int. J. Numer. Methods Eng. 20(4), 745–756. [doi:10.1002/nme.1620200412](https://doi.org/10.1002/nme.1620200412)
