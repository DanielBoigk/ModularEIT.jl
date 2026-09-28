# Adaptive vs uniform refinement (forward problem)

`benchmark/adaptive_meshing.jl`, 2026-09-28, Ryzen 7 7800X3D, single process, otherwise idle.
Every adaptive run is shown with every 4th step plus the last one; the full logs are in the
scratchpad archive.

**Setup.** Square [-1, 1]², complete electrode model with 16 electrodes (4 per side, width 0.25,
edges on mesh lines of every mesh, so the electrode geometry is identical on all meshes),
contact impedance 1e-3, 16 trigonometric current patterns. Q1 potential, piecewise constant σ.
Error: relative error of all electrode voltages against a uniform 1024 × 1024 reference
(1.05M unknowns, 11 s). Adaptive runs start at 16 × 16 and refine at most 6 levels (never below
the reference cell size); Dörfler marking with θ = 0.3; cells at the maximum level are excluded
from marking. Strategies: uniform, ZZ (flux recovery), GO (goal-oriented product indicator),
σ-jump (conductivity jumps only), GO+σ-jump (sum of normalised indicators).

## Scenario "circles": round inclusions sampled at cell centroids

The discrete inclusion shape changes with every mesh.

| strategy | unknowns | rel. voltage error | ZZ estimate | solve [s] | indicator [s] |
|:--|--:|--:|--:|--:|--:|
| uniform | 305 | 6.86e-02 | 2.41e+00 | 0.00 | 0.00 |
| uniform | 1105 | 3.56e-02 | 1.85e+00 | 0.01 | 0.00 |
| uniform | 4241 | 1.75e-02 | 1.28e+00 | 0.02 | 0.00 |
| uniform | 16657 | 8.08e-03 | 8.57e-01 | 0.12 | 0.00 |
| uniform | 66065 | 3.08e-03 | 5.59e-01 | 0.53 | 0.00 |
| uniform | 263185 | 8.93e-04 | 3.61e-01 | 1.82 | 0.00 |
| ZZ | 305 | 6.86e-02 | 2.41e+00 | 0.31 | 0.01 |
| ZZ | 655 | 6.96e-02 | 1.89e+00 | 0.01 | 0.02 |
| ZZ | 1685 | 5.18e-02 | 1.09e+00 | 0.03 | 0.06 |
| ZZ | 5554 | 2.53e-02 | 6.18e-01 | 0.06 | 0.25 |
| ZZ | 16082 | 1.71e-02 | 3.87e-01 | 0.15 | 0.77 |
| ZZ | 38519 | 1.39e-02 | 2.79e-01 | 0.32 | 1.56 |
| ZZ | 92905 | 1.27e-02 | 2.47e-01 | 1.03 | 3.85 |
| ZZ | 208918 | 9.77e-03 | 2.37e-01 | 2.18 | 8.86 |
| ZZ | 304251 | 9.04e-03 | 2.35e-01 | 3.02 | 13.48 |
| GO | 305 | 6.86e-02 | 2.41e+00 | 0.00 | 0.34 |
| GO | 705 | 7.13e-02 | 1.81e+00 | 0.01 | 0.05 |
| GO | 1834 | 5.40e-02 | 1.07e+00 | 0.04 | 0.17 |
| GO | 5920 | 2.54e-02 | 6.03e-01 | 0.07 | 0.53 |
| GO | 16783 | 1.70e-02 | 3.79e-01 | 0.17 | 1.53 |
| GO | 40120 | 1.32e-02 | 2.77e-01 | 0.50 | 3.31 |
| GO | 97426 | 1.26e-02 | 2.46e-01 | 1.07 | 8.33 |
| GO | 216923 | 1.45e-02 | 2.36e-01 | 2.07 | 18.87 |
| GO | 315291 | 9.87e-03 | 2.35e-01 | 3.08 | 28.09 |
| σ-jump | 305 | 6.86e-02 | 2.41e+00 | 0.00 | 0.06 |
| σ-jump | 441 | 7.15e-02 | 2.31e+00 | 0.00 | 0.00 |
| σ-jump | 900 | 7.42e-02 | 2.23e+00 | 0.03 | 0.00 |
| σ-jump | 2350 | 7.03e-02 | 2.17e+00 | 0.02 | 0.00 |
| σ-jump | 5086 | 7.05e-02 | 2.15e+00 | 0.08 | 0.00 |
| σ-jump | 6764 | 7.05e-02 | 2.14e+00 | 0.08 | 0.00 |
| σ-jump | 8402 | 7.03e-02 | 2.14e+00 | 0.09 | 0.00 |
| σ-jump | 9390 | 7.03e-02 | 2.14e+00 | 0.11 | 0.00 |
| σ-jump | 9799 | 7.03e-02 | 2.14e+00 | 0.09 | 0.00 |
| σ-jump | 9973 | 7.03e-02 | 2.14e+00 | 0.10 | 0.00 |
| σ-jump | 10012 | 7.03e-02 | 2.14e+00 | 0.10 | 0.00 |
| σ-jump | 10018 | 7.03e-02 | 2.14e+00 | 0.10 | 0.00 |
| GO+σ-jump | 305 | 6.86e-02 | 2.41e+00 | 0.00 | 0.08 |
| GO+σ-jump | 662 | 1.58e-01 | 2.00e+00 | 0.03 | 0.05 |
| GO+σ-jump | 1824 | 4.68e-02 | 1.32e+00 | 0.02 | 0.15 |
| GO+σ-jump | 5957 | 8.07e-03 | 7.62e-01 | 0.06 | 0.56 |
| GO+σ-jump | 12426 | 1.42e-02 | 4.71e-01 | 0.12 | 1.12 |
| GO+σ-jump | 24851 | 1.63e-02 | 3.17e-01 | 0.22 | 2.22 |
| GO+σ-jump | 56229 | 1.62e-02 | 2.60e-01 | 0.47 | 4.77 |
| GO+σ-jump | 136707 | 7.79e-03 | 2.41e-01 | 1.34 | 11.97 |
| GO+σ-jump | 294599 | 1.23e-02 | 2.35e-01 | 2.77 | 26.59 |
| GO+σ-jump | 346868 | 1.15e-02 | 2.35e-01 | 3.39 | 31.89 |

## Scenario "aligned": square inclusions made of base-mesh cells

σ is identical on every mesh; only the discretisation error of the potential remains.

| strategy | unknowns | rel. voltage error | ZZ estimate | solve [s] | indicator [s] |
|:--|--:|--:|--:|--:|--:|
| uniform | 305 | 7.11e-02 | 2.36e+00 | 0.00 | 0.00 |
| uniform | 1105 | 3.53e-02 | 1.81e+00 | 0.01 | 0.00 |
| uniform | 4241 | 1.69e-02 | 1.24e+00 | 0.02 | 0.00 |
| uniform | 16657 | 7.57e-03 | 8.30e-01 | 0.09 | 0.00 |
| uniform | 66065 | 2.98e-03 | 5.39e-01 | 0.49 | 0.00 |
| uniform | 263185 | 8.78e-04 | 3.45e-01 | 1.78 | 0.00 |
| ZZ | 305 | 7.11e-02 | 2.36e+00 | 0.32 | 0.01 |
| ZZ | 617 | 7.47e-02 | 1.83e+00 | 0.01 | 0.05 |
| ZZ | 1539 | 5.12e-02 | 1.05e+00 | 0.01 | 0.08 |
| ZZ | 5072 | 3.00e-02 | 5.80e-01 | 0.06 | 0.22 |
| ZZ | 15279 | 2.44e-02 | 3.59e-01 | 0.13 | 0.70 |
| ZZ | 37838 | 1.32e-02 | 2.65e-01 | 0.48 | 1.49 |
| ZZ | 92537 | 1.25e-02 | 2.33e-01 | 0.88 | 3.77 |
| ZZ | 212272 | 1.37e-02 | 2.23e-01 | 2.19 | 9.10 |
| ZZ | 307920 | 1.01e-02 | 2.22e-01 | 3.12 | 13.40 |
| GO | 305 | 7.11e-02 | 2.36e+00 | 0.00 | 0.35 |
| GO | 687 | 7.05e-02 | 1.71e+00 | 0.03 | 0.04 |
| GO | 1743 | 3.90e-02 | 9.91e-01 | 0.02 | 0.16 |
| GO | 5758 | 2.62e-02 | 5.50e-01 | 0.06 | 0.54 |
| GO | 17448 | 1.39e-02 | 3.40e-01 | 0.14 | 1.45 |
| GO | 42188 | 7.25e-03 | 2.59e-01 | 0.50 | 3.46 |
| GO | 104165 | 1.12e-02 | 2.31e-01 | 1.12 | 9.00 |
| GO | 234772 | 5.28e-03 | 2.23e-01 | 2.36 | 20.71 |
| GO | 335190 | 1.35e-02 | 2.21e-01 | 3.29 | 30.17 |
| σ-jump | 305 | 7.11e-02 | 2.36e+00 | 0.00 | 0.06 |
| σ-jump | 436 | 7.08e-02 | 2.27e+00 | 0.00 | 0.00 |
| σ-jump | 770 | 7.07e-02 | 2.22e+00 | 0.01 | 0.00 |
| σ-jump | 1522 | 7.03e-02 | 2.19e+00 | 0.01 | 0.00 |
| σ-jump | 3305 | 7.01e-02 | 2.16e+00 | 0.04 | 0.00 |
| σ-jump | 5207 | 7.01e-02 | 2.16e+00 | 0.07 | 0.00 |
| σ-jump | 5783 | 7.01e-02 | 2.16e+00 | 0.05 | 0.03 |
| σ-jump | 6422 | 7.01e-02 | 2.16e+00 | 0.06 | 0.00 |
| σ-jump | 7432 | 7.01e-02 | 2.16e+00 | 0.07 | 0.00 |
| σ-jump | 7730 | 7.01e-02 | 2.16e+00 | 0.07 | 0.00 |
| σ-jump | 7820 | 7.01e-02 | 2.16e+00 | 0.06 | 0.00 |
| σ-jump | 7841 | 7.01e-02 | 2.16e+00 | 0.08 | 0.00 |
| GO+σ-jump | 305 | 7.11e-02 | 2.36e+00 | 0.00 | 0.09 |
| GO+σ-jump | 607 | 8.60e-02 | 2.27e+00 | 0.01 | 0.07 |
| GO+σ-jump | 1415 | 6.62e-02 | 1.34e+00 | 0.03 | 0.11 |
| GO+σ-jump | 3607 | 3.10e-02 | 8.48e-01 | 0.05 | 0.33 |
| GO+σ-jump | 8694 | 1.21e-02 | 5.86e-01 | 0.09 | 0.79 |
| GO+σ-jump | 17092 | 1.66e-02 | 3.61e-01 | 0.15 | 1.48 |
| GO+σ-jump | 38851 | 1.31e-02 | 2.64e-01 | 0.31 | 3.30 |
| GO+σ-jump | 95581 | 1.58e-02 | 2.32e-01 | 0.88 | 8.58 |
| GO+σ-jump | 217669 | 1.36e-02 | 2.23e-01 | 2.06 | 19.56 |
| GO+σ-jump | 315225 | 9.30e-03 | 2.22e-01 | 3.10 | 28.74 |

## Diagnostics

**The constrained (hanging-node) space is correct.** Refinement nests conforming spaces, so the
discrete dissipated power Σ Iᵀ U must increase monotonically. It does (aligned scenario, GO
steps): 117.31 → 118.25 → 119.18 → 119.30 → 122.72 → 123.97 → 124.78 → 125.68, and the adaptive
mesh with 1311 unknowns matches the energy of the uniform 64² mesh (4241 unknowns, 125.71).

**Per pattern** (aligned scenario, GO adaptive mesh with ~15k unknowns vs uniform 128², 16.7k):

| pattern (k) | error uniform 128² | error GO adaptive |
|:--|--:|--:|
| cos θ, sin θ | 3.4e-2, 3.4e-2 | 7.0e-3, 8.3e-3 |
| cos 2θ, sin 2θ | 2.0e-2, 4.1e-2 | 2.1e-2, 1.8e-2 |
| k = 5 … 7 | 1.5e-2 – 2.9e-2 | 2.8e-2 – 1.2e-1 |

Normalising the indicator per pattern and per measurement (`normalize = true`) did not help:
with 25.7k cells the k ≥ 2 patterns were still worse than uniform 128².

## Findings

1. **Energy accuracy:** adaptive refinement works as expected. The ZZ estimate and the
   dissipated power converge with 3–13× fewer unknowns than uniform refinement.
2. **Electrode voltages:** for low-frequency patterns the goal-oriented indicator gives about 5×
   smaller errors than a uniform mesh of the same size. For high-frequency patterns the
   recovery-based indicators under-resolve the solution, and the total relative voltage error
   stagnates near 1e-2 for every adaptive strategy, while uniform refinement keeps converging
   (8.8e-4 at 263k unknowns). Recovery estimators are not reliable enough for this goal; a
   residual-based dual-weighted estimator (with the measurement duals that the Jacobian
   already computes) is the next thing to try.
3. **σ-jump refinement alone** does not improve the forward accuracy at all (7e-2, the 16²
   level): it never refines at the electrodes. It is a tool for the conductivity parametrisation,
   not for the forward model.
4. **Re-sampling σ** on every mesh ("circles") adds a geometry error that no forward indicator
   sees; for reconstructions the conductivity should be transferred (`transfer_conductivity`),
   not re-sampled.
5. **Cost:** the indicators cost more than the solves (GO at 300k unknowns: 28 s vs 3 s for
   the solve with 16 patterns; ZZ: 13 s). The L² projections of the recovery run on one core.

**Practical recommendation for now:** a uniform (or a priori graded towards the boundary) mesh
for the forward problem, and adaptivity for the conductivity parametrisation only.
