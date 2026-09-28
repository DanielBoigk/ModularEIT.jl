# Linear Solvers

```@meta
CurrentModule = ModularEIT
```

Projected block conjugate gradient method for the singular (pure Neumann) EIT systems
``L_\sigma X = B``. It works on the CPU and the GPU and supports several right-hand sides
at once. The theory is in the wiki articles *Projected Conjugate Gradient*,
*Block Conjugate Gradient* and *Grounding of the Potential*.

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

## GPU usage

```julia
using CUDA, CUDA.CUSPARSE, SparseArrays
to_device(x::SparseMatrixCSC) = CuSparseMatrixCSR(x)
to_device(x) = CuArray(x)

A_gpu = to_device(A)                     # A::SparseMatrixCSC{Float32}
B_gpu = CuArray(B)
M = AMGPreconditioner(A, size(B, 2); to_device)
X, stats = pbcg(A_gpu, B_gpu; M, grounding = boundary_grounding(size(A, 1), boundary_dofs))
```

Loading CUDA.jl activates the `ModularEITCUDAExt` package extension. It replaces the cuBLAS GEMM
used for the tall-skinny Gram products `XᵀY` by one GEMV per column for Float64 blocks with
2–8 columns. For those shapes cuBLAS picks a kernel that is 10–30× slower. Note that consumer
GPUs run Float64 at 1/64 of the Float32 rate, so Float32 is usually the better choice there
(see `benchmark/`).
