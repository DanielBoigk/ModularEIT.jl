---
tags: [numerics, fem, adaptivity]
aliases: [NVB coarsening, Bisection coarsening]
---

Coarsening a mesh produced by [[Newest Vertex Bisection]] (NVB) undoes bisections. Done correctly, it is the exact inverse of refinement: refining and coarsening back returns the same mesh, the mesh stays conforming, and no new triangle shapes appear.

## Which vertices can be removed

Every bisection creates one new vertex $m$, the midpoint of the refinement edge, and it is the *newest vertex* of the two children. When the refinement edge is interior, the neighbour across it is bisected at the same point, so $m$ is surrounded by four triangles; on the boundary by two. A vertex $m$ is **removable** if

1. $m$ is the newest vertex of *every* triangle in its star (the triangles containing $m$), and
2. the star has exactly four triangles (interior vertex) or two (boundary vertex).

Removing $m$ merges the star pairwise back into the two (or one) parents. Condition 1 guarantees that no triangle of the star has been bisected further at a different vertex, which would leave a hanging node. For meshes obtained by NVB from an initial mesh with compatible refinement edges, this *local* test is enough; no global refinement tree is needed.

## Reconstructing the parent without a tree

With the storage convention "triangle $(i,j,k)$, counter-clockwise, refinement edge $(i,j)$, newest vertex $k$", bisection at the midpoint $m$ of $(i,j)$ produces the children

$$
T_1 = (k,\,i,\,m),\qquad T_2 = (j,\,k,\,m).
$$

Both have $m$ as newest vertex, and the parent is recovered from them as

$$
(i,\,j,\,k) = \big(T_1[2],\ T_2[1],\ T_1[1]\big),
$$

with its refinement edge $(i,j)$ and newest vertex $k$ restored. The two children of a parent are the pair in the star of $m$ that share the edge $(k,m)$. Parent pointers, stored child indices or a forest are therefore not required. Only the storage convention must be kept.

## Algorithm

Given a set of triangles marked for coarsening:

1. **Candidate vertices.** Collect the newest vertices $m$ of marked triangles. Keep those whose whole star is marked and which satisfy the removability test. Stars of different removable vertices are disjoint, so all of them can be processed in one sweep.
2. **Merge.** For each vertex, pair the children across the edges $(k,m)$ and replace each pair by its parent (formula above). The parent's level is the children's level minus one.
3. **Edges.** The edge $(i,j)$ is whole again. Remove $m$ from the table *split edge → midpoint*. In every facet set that contains $(i,m)$ and $(m,j)$ (boundary edges, electrode edges), replace them by $(i,j)$.
4. **Cell data.** Cell sets are inherited from the children; they agree when the children came from one parent. Piecewise constant conductivities are transferred by area-weighted averaging, i.e. the [[L2 Projection]] onto the coarse mesh.
5. **Renumbering.** Delete the removed vertices from the node list and renumber the remaining nodes and all triangles, facet sets and midpoint entries.

One sweep coarsens by at most one level per vertex. Several levels need repeated sweeps, just as refinement is applied level by level. A marked triangle whose star is not fully marked is left alone, so coarsening never forces coarsening of unmarked cells. This mirrors how the refinement closure may force refinement of unmarked cells.

## Implementation notes

The bisection mesh (`BisectionMesh` in the AMR module of the Ferrite.jl fork, `src/Adaptivity/bisection.jl`) stores what the local algorithm needs: the triangles in the $(i,j,k)$ convention, their levels, the midpoint table (split edge → node), facet sets as node pairs and the leaves containing every edge. Coarsening (`coarsen!`) runs in two passes. The first finds the removable vertices: it inverts the midpoint table to get the edge each vertex split, and accepts a vertex only if the leaves of its star pair up into children whose reconstructed parent has exactly that edge as refinement edge. In a star of four this rejects the wrong pairing across the split edge. The second pass merges, restores the split edges in the midpoint table and the facet sets, and renumbers nodes and cells. Both passes are linear in the number of cells. Because coarsening renumbers the cells, refinement and coarsening marked on the same mesh are applied together (`refine_and_coarsen!`).

**In ModularEIT.jl:** [`coarsen_mesh!`](https://danielboigk.github.io/ModularEIT.jl/dev/api/adaptivity/#ModularEITFerrite.coarsen_mesh!).

## References

1. L. Chen, C.-S. Zhang (2010). *A Coarsening Algorithm on Adaptive Grids by Newest Vertex Bisection and Its Applications*. J. Comput. Math. 28(6), 767–789. [doi:10.4208/jcm.1004-m3172](https://doi.org/10.4208/jcm.1004-m3172)
2. S. A. Funken, D. Praetorius, P. Wissgott (2011). *Efficient implementation of adaptive P1-FEM in Matlab*. Comput. Methods Appl. Math. 11(4), 460–490. [doi:10.2478/cmam-2011-0026](https://doi.org/10.2478/cmam-2011-0026)
3. I. Kossaczký (1994). *A recursive approach to local mesh refinement in two and three dimensions*. J. Comput. Appl. Math. 55(3), 275–288. [doi:10.1016/0377-0427(94)90034-5](https://doi.org/10.1016/0377-0427(94)90034-5)
4. R. Stevenson (2008). *The completion of locally refined simplicial partitions created by bisection*. Math. Comput. 77(261), 227–241. [doi:10.1090/S0025-5718-07-01959-X](https://doi.org/10.1090/S0025-5718-07-01959-X)
