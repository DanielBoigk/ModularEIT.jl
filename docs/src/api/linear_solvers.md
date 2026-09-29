# Linear Solvers

```@meta
CurrentModule = ModularEIT
```

Projected block conjugate gradient method for the singular (pure Neumann) EIT systems
``L_\sigma X = B``. It works on the CPU and the GPU and supports several right-hand sides
at once. The theory is in the wiki articles [Projected Conjugate Gradient](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/04-Finite-Element-Discretization/Projected-Conjugate-Gradient),
[Block Conjugate Gradient](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/04-Finite-Element-Discretization/Block-Conjugate-Gradient) and [Grounding of the Potential](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/04-Finite-Element-Discretization/Grounding-of-the-Potential).

The objectives and forward solves take a solver *choice*, instantiated for the current system
matrix and refactorised / re-preconditioned when the conductivity changes:

```@docs
AbstractLinearSolver
DirectSolver
BlockCGSolver
```

## Projected block CG

```@docs
pbcg
pbcg!
BlockCGWorkspace
BlockCGStats
boundary_grounding
```

## Projected sparse Cholesky

Direct solver for the same systems: sparse Cholesky of the matrix with the null space pinned,
followed by the same grounding. CHOLMOD on the CPU (Float64), NVIDIA cuDSS on the GPU when
CUDA.jl and CUDSS.jl are loaded (`to_device` as below). `refactor!` reuses the symbolic
analysis when only the conductivity changes.

```@docs
projected_cholesky
projected_ldl
ProjectedCholesky
refactor!
LinearAlgebra.ldiv!(::AbstractVecOrMat, ::ProjectedCholesky, ::AbstractVecOrMat)
```

## Projected block MINRES (Krylov.jl)

Block MINRES from Krylov.jl on the projected system, with optional symmetric Jacobi scaling
(Krylov.jl's block MINRES does not take a preconditioner yet). `columnwise = true` uses the
single-vector MINRES per right-hand side, which also works on the GPU.

```@docs
pbminres
pbminres!
ProjectedMinresWorkspace
BlockMinresStats
```

## Preconditioners

```@docs
JacobiPreconditioner
AMGPreconditioner
ModularEIT.apply_preconditioner!
```

## Backend hooks

```@docs
ModularEIT._gram!
```

## GPU usage (any backend)

All solvers run on any GPUArrays.jl backend. `device_converter(ArrayType)` builds the `to_device`
function: sparse matrices become a [`DeviceSparseMatrixCSR`](@ref), whose products run through a
KernelAbstractions.jl kernel, and dense arrays become `ArrayType`. Only the `s × s` block
algebra and the AMG setup run on the CPU.

```julia
using AMDGPU                                     # or CUDA, oneAPI, Metal (Float32 only)
to_device = device_converter(ROCArray)           # CuArray, oneArray, MtlArray, ...

A_dev = to_device(A)                             # A::SparseMatrixCSC
B_dev = to_device(B)
M = AMGPreconditioner(A, size(B, 2); to_device)
w = boundary_grounding(size(A, 1), boundary_dofs)
X, stats = pbcg(A_dev, B_dev; M, grounding = w)

F = projected_cholesky(A; grounding = w, to_device)   # factorised on the host (see below)
X = F \ B_dev
```

The tests run this code path with JLArrays.jl, a CPU implementation of the GPUArrays interface
with scalar indexing disabled.

The direct solver factorises on the device only where a device factorisation exists: cuDSS on
NVIDIA GPUs, with `CuSparseMatrixCSR` matrices and CUDSS.jl loaded. On all other backends it
factorises on the host and copies right-hand sides and solutions. Krylov.jl's block MINRES
currently fails on GPUs; use `pbminres(...; columnwise = true)` there.

```@docs
device_converter
DeviceSparseMatrixCSR
```

### NVIDIA-specific extras

With CUDA.jl, `to_device = x -> x isa SparseMatrixCSC ? CuSparseMatrixCSR(x) : CuArray(x)`
uses cuSPARSE instead of the generic kernel and, together with CUDSS.jl, factorises on the GPU.
Loading CUDA.jl activates the `ModularEITCUDAExt` package extension. It replaces the cuBLAS GEMM
used for the tall-skinny Gram products `XᵀY` by one GEMV per column for Float64 blocks with
2–8 columns. For those shapes cuBLAS picks a kernel that is 10–30× slower. Note that consumer
GPUs run Float64 at 1/64 of the Float32 rate, so Float32 is usually the better choice there
(see `benchmark/`).

## DCT preconditioner (uniform rectangle grids)

On a uniform rectangle grid of bilinear elements (e.g. pixel-aligned image meshes) the
constant-conductivity system is inverted by fast cosine transforms. As a preconditioner, the
iteration count only depends on the conductivity contrast, not on the mesh:

```julia
solver = BlockCGSolver(preconditioner = DCTPreconditioner(disc))
obj = AdjointStateObjective(fm, currents, voltages; solver)
```

Theory: wiki articles [Discrete Cosine Transform](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/04-Finite-Element-Discretization/Discrete-Cosine-Transform) and [Fast Solvers on Rectangular Domains](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/04-Finite-Element-Discretization/Fast-Solvers-on-Rectangular-Domains).

```@docs
DCTPreconditioner
dct_preconditioner
update_preconditioner!
StructuredGrid
structured_grid
dct_neumann_solve
```

## FFT preconditioner (disk meshes)

On rotationally symmetric disk meshes of linear triangles ([`polar_grid`](@ref), with rings
graded towards the boundary if desired) the constant-conductivity system is inverted with an FFT
in the angle and tridiagonal solves along the radius. The iteration count again only depends on
the contrast; algebraic multigrid degrades on these anisotropic meshes.

```julia
disc = FerriteDiscretization(polar_grid(32, 256; boundary_spacing = 1 / 64))
solver = BlockCGSolver(preconditioner = PolarPreconditioner(disc))
```

Theory: wiki article [Fast Solvers on Disk Domains](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/04-Finite-Element-Discretization/Fast-Solvers-on-Disk-Domains).

```@docs
AbstractFastPreconditioner
PolarPreconditioner
polar_preconditioner
polar_grid
PolarStructure
polar_structure
fast_neumann_solve
```

## Wiki articles

Theory behind this page in the [theory wiki](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/):

- [Null Space of the Neumann Problem](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/04-Finite-Element-Discretization/Null-Space-of-the-Neumann-Problem)
- [Grounding of the Potential](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/04-Finite-Element-Discretization/Grounding-of-the-Potential)
- [Conjugate Gradient Method](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/04-Finite-Element-Discretization/Conjugate-Gradient-Method)
- [Projected Conjugate Gradient](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/04-Finite-Element-Discretization/Projected-Conjugate-Gradient)
- [Block Conjugate Gradient](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/04-Finite-Element-Discretization/Block-Conjugate-Gradient)
- [Block Krylov Methods](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/04-Finite-Element-Discretization/Block-Krylov-Methods)
- [Projected Cholesky Factorization](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/04-Finite-Element-Discretization/Projected-Cholesky-Factorization)
- [MINRES](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/04-Finite-Element-Discretization/MINRES)
- [Algebraic Multigrid](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/04-Finite-Element-Discretization/Algebraic-Multigrid)
- [Discrete Cosine Transform](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/04-Finite-Element-Discretization/Discrete-Cosine-Transform)
- [Fast Solvers on Rectangular Domains](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/04-Finite-Element-Discretization/Fast-Solvers-on-Rectangular-Domains)
- [Fast Solvers on Disk Domains](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/04-Finite-Element-Discretization/Fast-Solvers-on-Disk-Domains)
