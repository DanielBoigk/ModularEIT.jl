---
tags: [numerics, fem, adaptivity]
aliases: [Adaptive mesh refinement in EIT, AMR in EIT]
---

Adaptive mesh refinement repeats **SOLVE → ESTIMATE → MARK → REFINE** (see [[A Posteriori Error Estimation and Adaptive Meshing]]). In EIT it serves two different purposes, and they call for different indicators.

## 1. Accurate forward predictions

The discretisation error of the predicted voltages is a *model error*. If it is larger than the measurement noise, the reconstruction fits the discretisation error and produces artefacts (see [[Inverse Crime]] and [[Noise Models for EIT Data]]). The error concentrates where the solution is not smooth:

- **Electrode edges.** Under the [[Shunt Model]], the current density is singular like $r^{-1/2}$ at the electrode edges. Under the [[Complete Electrode Model]], a small contact impedance $z$ leaves a thin boundary layer with steep gradients instead. [[Point Electrode Model|Point electrodes]] have logarithmic singularities.
- **Conductivity interfaces.** At a jump of $\sigma$ the normal current $\sigma\partial_\nu u$ is continuous, so $\partial_\nu u$ jumps. The potential has a kink there that piecewise polynomials resolve only with small cells.
- **Near the boundary.** High-frequency current patterns decay quickly into the interior (see [[Current Patterns]]), so they need fine cells close to the boundary.
- **Reentrant corners**, if the domain has them. Convex corners, such as those of a square image domain, are harmless.

One mesh has to serve all current patterns, so the indicators of all patterns are summed. Suitable indicators:

- the [[Residual Estimator for the Conductivity Equation|residual indicator]]: element residuals, current-density jumps across facets, and the boundary residuals of the electrode model;
- the [[Zienkiewicz-Zhu Estimator]], which recovers a smooth current density and measures the distance to it;
- [[Goal-Oriented Error Estimation]], which weights the residuals by adjoint fields of the measurements and targets exactly the voltage error.

## 2. Resolving the conductivity

When $\sigma$ lives on the same mesh as $u$, refinement also adds conductivity unknowns. For the reconstruction, cells are needed where $\sigma$ has features: inclusion boundaries show up as large jumps of a piecewise constant iterate, i.e. large contributions to its [[Total Variation]]. Refining there and coarsening flat regions gives a feature-adapted parametrisation. Two cautions apply:

- **Resolution is limited by the data.** EIT is severely ill-posed (see [[Stability of the Calderón Problem]]). Interior details far from the electrodes cannot be resolved by any mesh. Refining there only adds unknowns that the [[Variational Regularization|regulariser]] must control.
- **Feature-driven refinement is blind to the forward error.** It does not refine at the electrodes. Combining both indicators, or using separate meshes for $u$ (fine) and $\sigma$ (coarse), keeps the model error small.

A convergent adaptive algorithm for the regularised EIT problem refines with an estimator that combines the state, the adjoint and the conductivity (Jin, Xu, Zou).

## Consequences for the algorithms

- **Gradients.** On a graded mesh the coefficient gradient $\partial J/\partial\sigma_a$ scales with the cell size, so steepest descent favours large cells. The $L^2$ gradient $M_\sigma^{-1}\partial J/\partial\sigma$ is mesh independent (see [[Conductivity Tensor]] and [[Gradient Representation and the Riesz Map]]).
- **Transfer.** After refinement or coarsening, the current iterate is transferred by [[L2 Projection]]. For piecewise constants, children inherit the value of their parent, and a coarsened parent gets the mean of its children.
- **The experiment must not change with the mesh.** Electrodes, current patterns and the grounding must be defined geometrically (for example electrode positions as length-weighted centroids, electrodes as boundary segments carried through refinement), not through node or facet counts, which change under boundary refinement. Otherwise every mesh simulates a slightly different measurement, and the discretisation error seems to stagnate.
- **Discretisation-dependent functionals.** Regularisers must be evaluated with the mesh geometry (cell areas, facet lengths), not per coefficient, or their weight changes under refinement.

## Refining quadrilaterals and triangles

Quadtree refinement of quadrilaterals (octrees in 3D) splits a cell into four children and creates [[Hanging Nodes]]. It fits pixel meshes well and keeps the cells shape regular. Triangles can be refined conformingly by [[Newest Vertex Bisection]] or red-green refinement, without hanging nodes. Marking is usually done by [[Dörfler Marking]].

## References

1. M. Molinari, B. H. Blott, S. J. Cox, G. J. Daniell (2002). *Optimal imaging with adaptive mesh refinement in electrical impedance tomography*. Physiol. Meas. 23(1), 121–128. [doi:10.1088/0967-3334/23/1/311](https://doi.org/10.1088/0967-3334/23/1/311)
2. B. Jin, Y. Xu, J. Zou (2017). *A convergent adaptive finite element method for electrical impedance tomography*. IMA J. Numer. Anal. 37(3), 1520–1550. [doi:10.1093/imanum/drw045](https://doi.org/10.1093/imanum/drw045)
3. R. Becker, B. Vexler (2004). *A posteriori error estimation for finite element discretization of parameter identification problems*. Numer. Math. 96(3), 435–459. [doi:10.1007/s00211-003-0482-9](https://doi.org/10.1007/s00211-003-0482-9)
4. M. Ainsworth, J. T. Oden (2000). *A Posteriori Error Estimation in Finite Element Analysis*. Wiley. [doi:10.1002/9781118032824](https://doi.org/10.1002/9781118032824)
