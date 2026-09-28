# Adaptive vs uniform refinement (forward problem)

`benchmark/adaptive_meshing.jl` (quadrilaterals, Ferrite AMR with hanging nodes) and
`benchmark/adaptive_meshing_triangles.jl` (triangles, newest vertex bisection), 2026-09-28,
Ryzen 7 7800X3D, single process, otherwise idle. Adaptive runs are shown with every 4th step
and the last one; full logs are in the scratchpad archive.

**Setup (quadrilaterals).** Square [-1, 1]², complete electrode model with 16 electrodes
(4 per side, width 0.25, edges on mesh lines of every mesh), contact impedance 1e-3, 16
trigonometric current patterns, Q1 potential, piecewise constant σ. Error: relative error of all
electrode voltages against a uniform 1024 × 1024 reference (1.05M unknowns, 10 s). Adaptive runs
start at 16 × 16 and refine at most 6 levels (the reference cell size). Dörfler marking,
θ = 0.3; cells at the maximum level are excluded from marking.

**Setup (triangles).** Unit disc (`test/circle.msh`, 2972 triangles), CEM with 16 electrodes
defined as facet sets of the base mesh, σ piecewise constant on the base mesh; both carried
exactly through bisection. Reference: base mesh bisected uniformly 8 times (381k unknowns);
adaptive runs refine at most 8 bisections.

Strategies: uniform; ZZ (flux recovery); RES (residual indicator); GO (goal-oriented, recovery
estimates of states × measurement duals); GO-res (the same with residual estimates); σ-jump
(conductivity jumps only); GO+σ-jump (sum of normalised indicators).

**Reading the numbers.** The adaptive meshes and the references have the same finest cells at
the singularities, so errors below ≈ 5e-4 measure agreement with the reference rather than the
true error (the reference itself differs from the next-coarser uniform mesh by ≈ 9e-4 on quads).
Speed-ups are therefore quoted at accuracies of ≈ 1e-3.

## Quadrilaterals, square inclusions made of base-mesh cells (σ identical on all meshes)

| strategy | unknowns | rel. voltage error | ZZ estimate | solve [s] | indicator [s] |
|:--|--:|--:|--:|--:|--:|
| uniform | 305 | 7.11e-02 | 2.36e+00 | 0.00 | 0.00 |
| uniform | 1105 | 3.53e-02 | 1.81e+00 | 0.01 | 0.00 |
| uniform | 4241 | 1.69e-02 | 1.24e+00 | 0.02 | 0.00 |
| uniform | 16657 | 7.57e-03 | 8.30e-01 | 0.23 | 0.00 |
| uniform | 66065 | 2.98e-03 | 5.39e-01 | 0.32 | 0.00 |
| uniform | 263185 | 8.78e-04 | 3.45e-01 | 1.60 | 0.00 |
| ZZ | 305 | 7.11e-02 | 2.36e+00 | 0.22 | 0.01 |
| ZZ | 638 | 2.68e-02 | 1.79e+00 | 0.01 | 0.02 |
| ZZ | 1580 | 7.35e-03 | 1.03e+00 | 0.01 | 0.05 |
| ZZ | 5377 | 1.53e-03 | 5.68e-01 | 0.05 | 0.24 |
| ZZ | 16033 | 4.51e-04 | 3.52e-01 | 0.13 | 0.65 |
| ZZ | 39551 | 1.36e-04 | 2.63e-01 | 0.26 | 1.67 |
| ZZ | 97155 | 3.60e-05 | 2.32e-01 | 0.82 | 3.94 |
| ZZ | 221436 | 9.27e-06 | 2.23e-01 | 2.06 | 9.55 |
| ZZ | 319022 | 4.84e-06 | 2.22e-01 | 2.91 | 13.60 |
| RES | 305 | 7.11e-02 | 2.36e+00 | 0.00 | 1.09 |
| RES | 487 | 4.33e-02 | 1.85e+00 | 0.00 | 0.00 |
| RES | 768 | 1.70e-02 | 1.56e+00 | 0.01 | 0.00 |
| RES | 1769 | 4.60e-03 | 1.06e+00 | 0.01 | 0.01 |
| RES | 5963 | 1.06e-03 | 6.64e-01 | 0.06 | 0.04 |
| RES | 19084 | 2.66e-04 | 4.52e-01 | 0.14 | 0.12 |
| RES | 55968 | 6.91e-05 | 3.47e-01 | 0.54 | 0.34 |
| RES | 146905 | 1.85e-05 | 2.95e-01 | 1.26 | 0.90 |
| RES | 322838 | 4.70e-06 | 2.55e-01 | 2.88 | 2.00 |
| GO | 305 | 7.11e-02 | 2.36e+00 | 0.00 | 0.27 |
| GO | 697 | 2.24e-02 | 1.70e+00 | 0.01 | 0.04 |
| GO | 1823 | 5.69e-03 | 9.67e-01 | 0.01 | 0.16 |
| GO | 5923 | 1.42e-03 | 5.44e-01 | 0.07 | 0.52 |
| GO | 17952 | 3.84e-04 | 3.37e-01 | 0.29 | 1.47 |
| GO | 43228 | 1.18e-04 | 2.58e-01 | 0.48 | 3.51 |
| GO | 106601 | 3.15e-05 | 2.31e-01 | 1.04 | 8.96 |
| GO | 239779 | 8.12e-06 | 2.23e-01 | 2.17 | 20.88 |
| GO | 340689 | 4.27e-06 | 2.22e-01 | 2.99 | 30.03 |
| GO-res | 305 | 7.11e-02 | 2.36e+00 | 0.00 | 0.06 |
| GO-res | 499 | 4.23e-02 | 1.83e+00 | 0.01 | 0.01 |
| GO-res | 788 | 1.59e-02 | 1.55e+00 | 0.01 | 0.01 |
| GO-res | 1938 | 4.07e-03 | 1.02e+00 | 0.02 | 0.03 |
| GO-res | 6444 | 9.79e-04 | 6.52e-01 | 0.07 | 0.09 |
| GO-res | 20753 | 2.42e-04 | 4.42e-01 | 0.15 | 0.29 |
| GO-res | 60588 | 6.22e-05 | 3.37e-01 | 0.40 | 0.82 |
| GO-res | 156593 | 1.67e-05 | 2.91e-01 | 1.35 | 2.14 |
| GO-res | 338576 | 4.33e-06 | 2.55e-01 | 3.01 | 4.71 |
| σ-jump | 305 | 7.11e-02 | 2.36e+00 | 0.00 | 0.02 |
| σ-jump | 436 | 7.08e-02 | 2.27e+00 | 0.00 | 0.00 |
| σ-jump | 770 | 7.07e-02 | 2.22e+00 | 0.01 | 0.00 |
| σ-jump | 1522 | 7.03e-02 | 2.19e+00 | 0.01 | 0.00 |
| σ-jump | 3305 | 7.01e-02 | 2.16e+00 | 0.02 | 0.00 |
| σ-jump | 5207 | 7.01e-02 | 2.16e+00 | 0.04 | 0.00 |
| σ-jump | 5783 | 7.01e-02 | 2.16e+00 | 0.07 | 0.00 |
| σ-jump | 6422 | 7.01e-02 | 2.16e+00 | 0.07 | 0.00 |
| σ-jump | 7432 | 7.01e-02 | 2.16e+00 | 0.06 | 0.00 |
| σ-jump | 7730 | 7.01e-02 | 2.16e+00 | 0.08 | 0.00 |
| σ-jump | 7820 | 7.01e-02 | 2.16e+00 | 0.08 | 0.00 |
| σ-jump | 7841 | 7.01e-02 | 2.16e+00 | 0.09 | 0.00 |
| GO+σ-jump | 305 | 7.11e-02 | 2.36e+00 | 0.00 | 0.08 |
| GO+σ-jump | 607 | 6.35e-02 | 2.27e+00 | 0.00 | 0.04 |
| GO+σ-jump | 1409 | 1.53e-02 | 1.35e+00 | 0.01 | 0.10 |
| GO+σ-jump | 3587 | 5.07e-03 | 8.48e-01 | 0.03 | 0.32 |
| GO+σ-jump | 8653 | 2.07e-03 | 5.89e-01 | 0.08 | 0.74 |
| GO+σ-jump | 17007 | 5.59e-04 | 3.63e-01 | 0.14 | 1.60 |
| GO+σ-jump | 38586 | 1.41e-04 | 2.64e-01 | 0.26 | 3.28 |
| GO+σ-jump | 94727 | 3.73e-05 | 2.33e-01 | 0.77 | 8.42 |
| GO+σ-jump | 215998 | 9.67e-06 | 2.23e-01 | 1.88 | 18.98 |
| GO+σ-jump | 313199 | 5.03e-06 | 2.22e-01 | 2.63 | 28.20 |

## Quadrilaterals, round inclusions sampled at the cell centroids of each mesh

| strategy | unknowns | rel. voltage error | ZZ estimate | solve [s] | indicator [s] |
|:--|--:|--:|--:|--:|--:|
| uniform | 305 | 6.86e-02 | 2.41e+00 | 0.00 | 0.00 |
| uniform | 1105 | 3.56e-02 | 1.85e+00 | 0.01 | 0.00 |
| uniform | 4241 | 1.75e-02 | 1.28e+00 | 0.02 | 0.00 |
| uniform | 16657 | 8.08e-03 | 8.57e-01 | 0.08 | 0.00 |
| uniform | 66065 | 3.08e-03 | 5.59e-01 | 0.44 | 0.00 |
| uniform | 263185 | 8.93e-04 | 3.61e-01 | 1.57 | 0.00 |
| RES | 305 | 6.86e-02 | 2.41e+00 | 0.22 | 1.15 |
| RES | 487 | 4.11e-02 | 1.91e+00 | 0.03 | 0.00 |
| RES | 778 | 1.48e-02 | 1.63e+00 | 0.01 | 0.01 |
| RES | 1805 | 5.39e-03 | 1.12e+00 | 0.03 | 0.01 |
| RES | 6016 | 1.20e-03 | 6.71e-01 | 0.08 | 0.05 |
| RES | 16251 | 5.69e-04 | 4.05e-01 | 0.13 | 0.13 |
| RES | 43258 | 2.92e-04 | 2.97e-01 | 0.48 | 0.34 |
| RES | 106084 | 1.19e-04 | 2.56e-01 | 1.02 | 0.76 |
| RES | 235221 | 1.31e-04 | 2.41e-01 | 2.14 | 1.59 |
| RES | 336885 | 1.31e-04 | 2.38e-01 | 3.17 | 2.32 |
| GO-res | 305 | 6.86e-02 | 2.41e+00 | 0.00 | 0.25 |
| GO-res | 502 | 4.00e-02 | 1.89e+00 | 0.01 | 0.01 |
| GO-res | 798 | 1.39e-02 | 1.62e+00 | 0.01 | 0.01 |
| GO-res | 1852 | 4.45e-03 | 1.04e+00 | 0.01 | 0.03 |
| GO-res | 6238 | 1.15e-03 | 6.60e-01 | 0.05 | 0.10 |
| GO-res | 16658 | 5.55e-04 | 4.04e-01 | 0.13 | 0.27 |
| GO-res | 43716 | 2.54e-04 | 2.94e-01 | 0.32 | 0.68 |
| GO-res | 107996 | 1.22e-04 | 2.56e-01 | 1.04 | 1.67 |
| GO-res | 240263 | 1.31e-04 | 2.40e-01 | 2.16 | 3.59 |
| GO-res | 341773 | 1.31e-04 | 2.38e-01 | 3.01 | 5.16 |
| σ-jump | 305 | 6.86e-02 | 2.41e+00 | 0.00 | 0.02 |
| σ-jump | 441 | 7.15e-02 | 2.31e+00 | 0.00 | 0.00 |
| σ-jump | 900 | 7.42e-02 | 2.23e+00 | 0.01 | 0.00 |
| σ-jump | 2350 | 7.03e-02 | 2.17e+00 | 0.02 | 0.00 |
| σ-jump | 5086 | 7.05e-02 | 2.15e+00 | 0.06 | 0.00 |
| σ-jump | 6764 | 7.05e-02 | 2.14e+00 | 0.08 | 0.00 |
| σ-jump | 8402 | 7.03e-02 | 2.14e+00 | 0.08 | 0.00 |
| σ-jump | 9390 | 7.03e-02 | 2.14e+00 | 0.12 | 0.00 |
| σ-jump | 9799 | 7.03e-02 | 2.14e+00 | 0.09 | 0.00 |
| σ-jump | 9973 | 7.03e-02 | 2.14e+00 | 0.08 | 0.03 |
| σ-jump | 10012 | 7.03e-02 | 2.14e+00 | 0.11 | 0.00 |
| σ-jump | 10018 | 7.03e-02 | 2.14e+00 | 0.10 | 0.00 |

## Triangles (newest vertex bisection), unit disc

| strategy | unknowns | rel. voltage error | solve [s] | indicator [s] |
|:--|--:|--:|--:|--:|
| uniform | 1566 | 4.57e-02 | 0.02 | 0.00 |
| uniform | 3652 | 3.45e-02 | 0.04 | 0.00 |
| uniform | 8156 | 2.12e-02 | 0.09 | 0.00 |
| uniform | 17720 | 1.39e-02 | 0.18 | 0.00 |
| uniform | 37919 | 8.24e-03 | 0.39 | 0.00 |
| uniform | 79858 | 4.81e-03 | 0.94 | 0.00 |
| uniform | 165114 | 2.27e-03 | 2.05 | 0.00 |
| RES | 1566 | 4.57e-02 | 0.02 | 1.11 |
| RES | 1705 | 3.34e-02 | 0.02 | 0.01 |
| RES | 2076 | 1.47e-02 | 0.02 | 0.01 |
| RES | 3569 | 6.09e-03 | 0.04 | 0.02 |
| RES | 7620 | 2.50e-03 | 0.08 | 0.05 |
| RES | 16324 | 1.01e-03 | 0.16 | 0.12 |
| RES | 33385 | 3.90e-04 | 0.49 | 0.25 |
| RES | 64945 | 1.45e-04 | 0.73 | 0.49 |
| RES | 116918 | 4.92e-05 | 1.59 | 0.91 |
| RES | 185216 | 1.50e-05 | 2.58 | 1.49 |
| RES | 202697 | 1.13e-05 | 2.89 | 1.71 |
| GO-res | 1566 | 4.57e-02 | 0.02 | 1.37 |
| GO-res | 1705 | 3.34e-02 | 0.02 | 0.03 |
| GO-res | 2076 | 1.47e-02 | 0.05 | 0.03 |
| GO-res | 3561 | 6.12e-03 | 0.05 | 0.05 |
| GO-res | 7591 | 2.51e-03 | 0.07 | 0.12 |
| GO-res | 16264 | 1.02e-03 | 0.16 | 0.27 |
| GO-res | 33266 | 3.92e-04 | 0.53 | 0.56 |
| GO-res | 64669 | 1.46e-04 | 0.70 | 1.28 |
| GO-res | 116441 | 4.96e-05 | 1.42 | 2.26 |
| GO-res | 184661 | 1.51e-05 | 2.60 | 3.47 |
| GO-res | 202308 | 1.13e-05 | 2.79 | 3.86 |

## Findings

1. **Adaptivity pays off.** For about 1e-3 voltage accuracy, the adaptive quadrilateral meshes
   need about 6k unknowns and the uniform ones about 250k (≈ 40×). On triangles: 2.5e-3 with
   7.6k adaptive unknowns vs 2.3e-3 with 165k uniform ones (≈ 20×). The error is dominated by
   the electrode edges; all forward-error indicators find them.
2. **Indicator choice.** ZZ, RES, GO and GO-res give nearly the same meshes and errors here
   (for smooth-ish σ the energy error and the voltage error are driven by the same singularities).
   The residual indicators are much cheaper: RES 2 s and GO-res 5 s at 330k unknowns vs 14 s
   (ZZ) and 30 s (GO), i.e. about the cost of one forward solve with 16 patterns.
3. **σ-jump refinement alone** does not improve the forward accuracy (7e-2, the level of the
   16² mesh): it never refines at the electrodes. It serves the conductivity parametrisation.
4. **Re-sampling σ** on every mesh ("round inclusions") adds a geometry error that no forward
   indicator sees: the adaptive error levels off at 1.3e-4 (still 7× below uniform at 263k).
   In reconstructions σ is transferred between meshes (`transfer_conductivity`), not re-sampled.
5. **Hanging nodes are consistent**: refining one cell and then everything uniformly reproduces
   the uniform errors (6.3e-3 vs 6.5e-3 after three uniform steps), and the discrete dissipated
   power increases monotonically under adaptive refinement.

## A bug found through this comparison

A first version of this study showed all adaptive strategies stagnating near 1e-2. The cause
was not the refinement but the *current patterns*: electrode angles were computed from plain
averages of facet midpoints and boundary-node coordinates, which shift when boundary cells are
refined, so `trigonometric_patterns` generated slightly different patterns on every mesh.
Angles now use length-weighted boundary centroids and are mesh independent (regression test in
`test/test_adaptive_meshing.jl`). Remaining mesh dependence: the gap/point/continuum models
ground voltages by the *nodal* boundary sum, which shifts all voltages by a constant when the
boundary node distribution changes; the objectives remove the mean, and comparisons should too.
