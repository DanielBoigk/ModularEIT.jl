# ModularEIT.jl

A Julia library for **Electrical Impedance Tomography (EIT)** built from exchangeable parts:
finite element discretizations, electrode models, forward solvers, objectives with adjoint
gradients, regularizers, optimizers, fast linear solvers, and synthetic data. Every part can be
replaced without touching the others.

- [API documentation](https://danielboigk.github.io/ModularEIT.jl/dev/)
- [Theory wiki](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/): the physics, mathematics
  and numerics behind the library, with references. Articles on implemented topics link to the
  API documentation and vice versa.

> **Status:** under active development and not yet registered. The API may still change. An
> earlier implementation is kept in the other branches of this repository.

## Features

- **Discretization** through back end packages, `ModularEITFerrite`
  ([Ferrite.jl](https://github.com/Ferrite-FEM/Ferrite.jl)) and `ModularEITGridap`
  ([Gridap.jl](https://github.com/gridap/Gridap.jl)): separate finite
  element spaces for potential and conductivity (e.g. P1/P0, Q1/Q0 on pixel images), triangles and
  quadrilaterals, and a conductivity tensor that makes the system matrix linear in σ.
- **Electrode models:** continuum, point, gap and complete electrode model, with current-driven
  and voltage-driven forward problems.
- **Objectives:** least squares with adjoint-state gradients and Jacobians (weighted misfits,
  L² or coefficient gradients), and the Kohn–Vogelius functional.
- **Regularization and optimization:**
  - regularizers: Tikhonov (L², H¹, jump penalty) and total variation, smoothed or exact;
  - methods: Gauss–Newton / Levenberg–Marquardt, L-BFGS and gradient descent with bounds,
    proximal gradient (FISTA) and ADMM with exact TV proximal maps or user-defined denoisers
    (plug-and-play).
- **Linear solvers:** projected sparse Cholesky / LDLᵀ and projected block CG / MINRES for the
  singular Neumann systems, with preconditioners:
  - Jacobi and algebraic multigrid;
  - FFT-based preconditioners that are exact for constant conductivity: DCT on rectangles, polar
    FFT on disk meshes, and via conformal maps (Theodorsen's and Wegmann's methods) on other
    planar domains.

  Block CG with Jacobi or AMG runs on the CPU and on GPUs (KernelAbstractions); the Cholesky
  factorization uses cuDSS on NVIDIA GPUs (cuDSS extension). The FFT preconditioners
  currently run on the CPU.
- **Meshes:** adaptive refinement (residual, recovery-based and goal-oriented estimators, Dörfler
  marking; hanging nodes on quadrilaterals, newest vertex bisection with coarsening on triangles), graded polar
  meshes of the disk, and conformally mapped meshes.
- **Data:** noise models (absolute, relative, source/meter, operator-level, electrode modelling
  errors), mesh-independent phantoms (random inclusions, Gaussian random fields, images),
  simulation on finer meshes without inverse crime, discrepancy-principle targets, and
  image ↔ finite element maps.

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/DanielBoigk/Ferrite.jl", rev = "adaptive-triangular")
Pkg.add(url = "https://github.com/DanielBoigk/Krylov.jl", rev = "block-cg")
Pkg.add(url = "https://github.com/DanielBoigk/ModularEIT.jl")
Pkg.add(url = "https://github.com/DanielBoigk/ModularEIT.jl", subdir = "lib/ModularEITFerrite")
# or, for the Gridap back end (no Ferrite fork needed):
Pkg.add(url = "https://github.com/DanielBoigk/ModularEIT.jl", subdir = "lib/ModularEITGridap")
```

ModularEIT has no finite element code of its own; the discretization comes from a back end
package in `lib/` of this repository: `ModularEITFerrite`
([Ferrite.jl](https://github.com/Ferrite-FEM/Ferrite.jl)) or `ModularEITGridap`
([Gridap.jl](https://github.com/gridap/Gridap.jl)). Load ModularEIT with one of them, e.g.
`using ModularEIT, ModularEITFerrite`.

ModularEIT depends on a fork of [Krylov.jl](https://github.com/DanielBoigk/Krylov.jl) (block
conjugate gradients with null-space projection), and the Ferrite back end on a fork of
[Ferrite.jl](https://github.com/DanielBoigk/Ferrite.jl) (newest vertex bisection with coarsening
for triangle meshes). Add the forks first: Julia uses the `[sources]` entries of a package only
when it is the active project, so `Pkg.add` alone would install the registered versions, and
ModularEIT would fail to load.

## Example

Reconstruct an inclusion from noisy complete-electrode-model data. The data are simulated on a
finer mesh with the same electrodes to avoid the inverse crime.

```julia
using ModularEIT, ModularEITFerrite, Ferrite, Random

# reconstruction mesh (disk, rings graded towards the boundary) and 16-electrode CEM
disc = FerriteDiscretization(polar_grid(16, 128; boundary_spacing = 1 / 40))
els  = angular_electrodes(disc, 16; coverage = 0.5)
fm   = ForwardModel(disc, CompleteElectrodeModel(els, 0.05))

# simulated data on a finer mesh with the same electrodes, 1 % noise
fine     = FerriteDiscretization(polar_grid(32, 256; boundary_spacing = 1 / 80))
fm_fine  = ForwardModel(fine, CompleteElectrodeModel(transfer_electrodes(disc, els, fine), 0.05))
phantom  = InclusionPhantom(1.0, [CircleInclusion((0.3, 0.2), 0.3, 3.0)])
currents = trigonometric_patterns(fm, 7)
noise    = RelativeGaussianNoise(0.01)
sim      = simulate_data(fine, fm_fine, phantom, currents; noise, rng = MersenneTwister(1))

# TV-regularized least squares with the FFT-preconditioned solver, Gauss–Newton,
# stopped by the discrepancy principle
solver = BlockCGSolver(preconditioner = PolarPreconditioner(disc))
data   = AdjointStateObjective(fm, currents, sim.data; solver)
obj    = RegularizedObjective(data, 1e-4 => TotalVariationRegularizer(disc; ε = 1e-2))
res    = minimize(obj, ones(ndofs_σ(disc)), GaussNewton(); lower = 0.05,
                  ftarget = discrepancy_target(data, noise))
res.σ                       # reconstructed conductivity (one value per cell)
```

A worked version with plots is the tutorial
[Reconstructing a conductivity](https://danielboigk.github.io/ModularEIT.jl/dev/tutorials/reconstruction/)
(also as a Jupyter notebook). The showcase
[A landscape as conductivity](https://danielboigk.github.io/ModularEIT.jl/dev/tutorials/showcase_landscape/)
reconstructs a photograph from 32-electrode data: pixels as unknowns, the boundary-data SVD at
the noise level, sensitivity-damped Levenberg–Marquardt and a map of what the data determine.
The outlook
[toy plasma reactor](https://danielboigk.github.io/ModularEIT.jl/dev/tutorials/toy_plasma_reactor/)
uses the same machinery (conductivity equation, adjoint gradients) for topology optimisation
of a flow channel around a plasma arc.
More in
[Getting Started](https://danielboigk.github.io/ModularEIT.jl/dev/getting_started/) and the
[API documentation](https://danielboigk.github.io/ModularEIT.jl/dev/).

Tutorials are [Literate.jl](https://github.com/fredrikekre/Literate.jl) scripts in `examples/`;
the documentation build runs them and generates the pages and notebooks.

## Repository layout

| Path | Contents |
|:--|:--|
| `src/` | The library: `LinearSolvers/`, `Galerkin/` (back end contract, electrode models, forward model, objectives, regularizers), `Optimization/`, `Data/`, `Geometry/` |
| `lib/ModularEITFerrite/` | The Ferrite.jl back end (its own package and tests) |
| `lib/ModularEITGridap/` | The Gridap.jl back end (its own package; its tests compare it with the Ferrite back end) |
| `ext/` | cuDSS extension (GPU factorization) |
| `test/` | Test suite of the generic layer (`julia --project -e 'using Pkg; Pkg.test()'`); the back end's tests: `julia --project=lib/ModularEITFerrite -e 'using Pkg; Pkg.test()'` |
| `benchmark/` | Benchmark scripts and results |
| `examples/` | Tutorials as Literate.jl scripts (rendered into the documentation and notebooks) |
| `docs/` | Documenter.jl API documentation; the build also renders the wiki and checks the links between both |
| `markdown/` | The theory wiki (an Obsidian vault) |
| `site/` | Quartz, which renders the wiki as a website |

## License

MIT, see [LICENSE.md](LICENSE.md).
