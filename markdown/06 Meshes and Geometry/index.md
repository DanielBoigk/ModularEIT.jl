---
title: Meshes and Geometry
tags: [overview, fem, adaptivity, geometry]
---

Where the mesh has to be fine, and how to build it: a posteriori error estimation and adaptive refinement, and conformal maps that carry disk meshes to other domains.

**Reading order.**

1. Error estimation: [[A Posteriori Error Estimation and Adaptive Meshing]], with the [[Residual Estimator for the Conductivity Equation]], the [[Zienkiewicz-Zhu Estimator]] and [[Goal-Oriented Error Estimation]] for the measured quantities.
2. Refinement: [[Dörfler Marking]], [[Hanging Nodes]] on quadrilaterals, [[Newest Vertex Bisection]] on triangles, and [[Coarsening of Bisection Meshes]].
3. In EIT: [[Adaptive Meshing in EIT]].
4. Geometry: [[Numerical Conformal Mapping]], based on the [[Conformal Invariance of the Conductivity Equation]].

**Related.** Fast solvers for the mapped meshes: [[05 Linear Solvers/index|Linear Solvers]]. Transferring conductivities between meshes: [[L2 Projection]].

## References

1. R. Verfürth (2013). *A Posteriori Error Estimation Techniques for Finite Element Methods*. Oxford University Press. [doi:10.1093/acprof:oso/9780199679423.001.0001](https://doi.org/10.1093/acprof:oso/9780199679423.001.0001)
