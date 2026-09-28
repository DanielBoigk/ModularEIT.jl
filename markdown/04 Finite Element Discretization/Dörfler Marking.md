---
tags: [numerics, fem, adaptivity]
aliases: [Bulk criterion, Dorfler marking]
---

**Dörfler (bulk) marking** decides which cells to refine from the local error indicators $\eta_K$. Given $\theta\in(0,1]$, mark a set $\mathcal M$ of cells with

$$
\sum_{K\in\mathcal M}\eta_K^2\ \ge\ \theta\sum_K\eta_K^2 ,
$$

and choose $\mathcal M$ as small as possible: sort the cells by decreasing $\eta_K$ and take them until the fraction $\theta$ of the total estimated error is reached.

**Why this criterion.** Refining a fixed *fraction of the error*, rather than a fixed fraction of the cells or all cells above a threshold, guarantees a contraction of the error in each step of the adaptive loop. With a minimal marked set, the error decays with the best possible rate in terms of the number of unknowns. For singular solutions, such as currents at electrode edges, this rate is better than that of uniform refinement.

**Choice of $\theta$.** A small $\theta$ (0.2–0.5) refines few cells per step. The meshes are nearly optimal, but many solve–estimate cycles are needed. A large $\theta$ approaches uniform refinement. With several goals, for example forward accuracy and conductivity features in EIT (see [[Adaptive Meshing in EIT]]), the indicators can be combined before marking, or marked separately and the marked sets merged.

## References

1. W. Dörfler (1996). *A Convergent Adaptive Algorithm for Poisson's Equation*. SIAM J. Numer. Anal. 33(3), 1106–1124. [doi:10.1137/0733054](https://doi.org/10.1137/0733054)
2. R. Stevenson (2007). *Optimality of a Standard Adaptive Finite Element Method*. Found. Comput. Math. 7(2), 245–269. [doi:10.1007/s10208-005-0183-0](https://doi.org/10.1007/s10208-005-0183-0)
3. J. M. Cascón, C. Kreuzer, R. H. Nochetto, K. G. Siebert (2008). *Quasi-Optimal Convergence Rate for an Adaptive Finite Element Method*. SIAM J. Numer. Anal. 46(5), 2524–2550. [doi:10.1137/07069047X](https://doi.org/10.1137/07069047X)
