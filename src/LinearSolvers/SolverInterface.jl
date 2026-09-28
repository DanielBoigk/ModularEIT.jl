# Swappable linear solvers for the forward model and the objectives. A solver choice
# (DirectSolver, BlockCGSolver) is instantiated for one matrix (`_init_solver`), updated when the
# matrix values change (`_update_solver!`, same sparsity pattern) and applied to blocks of
# right-hand sides (`_solve!`). Singular (pure Neumann / CEM) systems pass their null space and
# grounding; nonsingular (Dirichlet) systems pass `n × 0` matrices.

"""
    DirectSolver(; backend = :auto)

Projected sparse Cholesky factorisation ([`projected_cholesky`](@ref)); refactorised numerically
when the conductivity changes. `backend`: `:auto`, `:cholmod` or `:ldl`.
"""
struct DirectSolver <: AbstractLinearSolver
    backend::Symbol
end
DirectSolver(; backend::Symbol = :auto) = DirectSolver(backend)

"""
    BlockCGSolver(; preconditioner = :amg, rtol = 1e-10, maxiter = 0)

Projected block conjugate gradients ([`pbcg!`](@ref)) with an `:amg`, `:jacobi` or `:none`
preconditioner. `maxiter = 0` uses the default of `pbcg!`. The previous solution is the initial
guess of the next solve (warm start across conductivity updates).
"""
struct BlockCGSolver <: AbstractLinearSolver
    preconditioner::Symbol
    rtol::Float64
    maxiter::Int
    function BlockCGSolver(preconditioner::Symbol, rtol::Real, maxiter::Integer)
        preconditioner in (:amg, :jacobi, :none) ||
            throw(ArgumentError("preconditioner must be :amg, :jacobi or :none, got :$preconditioner"))
        return new(preconditioner, rtol, maxiter)
    end
end
BlockCGSolver(; preconditioner::Symbol = :amg, rtol::Real = 1e-10, maxiter::Integer = 0) =
    BlockCGSolver(preconditioner, rtol, maxiter)

mutable struct _DirectState{F}
    F::F
end

function _init_solver(s::DirectSolver, A::SparseMatrixCSC, nullspace, grounding)
    return _DirectState(projected_cholesky(A; nullspace, grounding, backend = s.backend))
end
_update_solver!(st::_DirectState, A::SparseMatrixCSC) = (refactor!(st.F, A); st)
_solve!(X, st::_DirectState, B) = ldiv!(X, st.F, B)

mutable struct _CGState{MA}
    choice::BlockCGSolver
    A::MA
    nullspace::Matrix{Float64}
    grounding::Matrix{Float64}
    workspaces::Dict{Int, Any}        # per block size
    preconditioners::Dict{Int, Any}   # per block size, rebuilt after matrix updates
end

function _init_solver(s::BlockCGSolver, A::SparseMatrixCSC, nullspace, grounding)
    return _CGState(s, A, Matrix{Float64}(_as_matrix(nullspace)), Matrix{Float64}(_as_matrix(grounding)),
                    Dict{Int, Any}(), Dict{Int, Any}())
end

function _update_solver!(st::_CGState, A::SparseMatrixCSC)
    st.A = A
    empty!(st.preconditioners)
    return st
end

function _cg_preconditioner(st::_CGState, s::Integer)
    return get!(st.preconditioners, s) do
        pc = st.choice.preconditioner
        pc === :amg ? AMGPreconditioner(st.A, s) : pc === :jacobi ? JacobiPreconditioner(st.A) : nothing
    end
end

function _solve!(X, st::_CGState, B)
    s = size(_as_matrix(B), 2)
    ws = get!(() -> BlockCGWorkspace(st.A, _as_matrix(B); nullspace = st.nullspace, grounding = st.grounding),
              st.workspaces, s)
    kw = st.choice.maxiter > 0 ? (; maxiter = st.choice.maxiter) : (;)
    stats = pbcg!(X, ws, st.A, B; M = _cg_preconditioner(st, s), rtol = st.choice.rtol, kw...)
    stats.converged || @warn "block CG did not converge in $(stats.iterations) iterations"
    return X
end
