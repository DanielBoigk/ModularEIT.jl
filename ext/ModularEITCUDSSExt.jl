module ModularEITCUDSSExt

# GPU backend for `projected_cholesky`: sparse Cholesky (LLᵀ) factorisation and block solves
# with NVIDIA cuDSS. Activated when CUDA.jl and CUDSS.jl are loaded.

using ModularEIT
using CUDA
using CUDA.CUSPARSE
using CUDSS
using LinearAlgebra

struct CudssCholesky{T, S}
    solver::S
    x::CuVector{T}        # dummy vectors for the analysis / (re)factorisation phases
    b::CuVector{T}
end

function ModularEIT._chol_factorize(A::CuSparseMatrixCSR{T}) where {T <: Union{Float32, Float64}}
    solver = CudssSolver(A, "SPD", 'L')
    x, b = CUDA.zeros(T, size(A, 1)), CUDA.zeros(T, size(A, 1))
    cudss("analysis", solver, x, b)                 # ordering + symbolic factorisation
    cudss("factorization", solver, x, b)
    return CudssCholesky(solver, x, b)
end

# The reduced matrix is symmetric, so its CSR arrays coincide with the CSC arrays of the host
# copy: new values can be copied over directly.
ModularEIT._set_values!(A::CuSparseMatrixCSR, nzval::Vector) = (copyto!(A.nzVal, nzval); A)

function ModularEIT._chol_refactor!(F::CudssCholesky, ::CuSparseMatrixCSR)
    cudss("refactorization", F.solver, F.x, F.b)    # reuses the symbolic analysis
    return F
end

ModularEIT._chol_solve!(X::CuMatrix, F::CudssCholesky, B::CuMatrix) = (cudss("solve", F.solver, X, B); X)

end
