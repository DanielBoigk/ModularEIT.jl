# Discrete EIT forward model at the matrix level (independent of the finite element package).
#
# Current-driven (Neumann) problem:      A(σ) x = P I,    measured voltages  V = Q x,
#     A singular with null space `nullspace`, solution grounded by `groundingᵀ x = 0`.
# Voltage-driven (Dirichlet) problem:    x_B = E U prescribed on the Dirichlet dofs B,
#     A_II x_I = -A_IB x_B on the free dofs I,   currents  I = C⁻¹ Eᵀ (A x)_B,
#     with C = Eᵀ P_B, so that currents are returned in the same representation that P takes
#     (densities for the continuum model, electrode currents otherwise). For consistent data
#     the two problems are inverse to each other.
#
# A(σ) = A₀ + L(σ): A₀ holds constant contributions (complete electrode model), L(σ) comes from
# the ConductivityTensor, whose pattern is the pattern of A.

"""
    ForwardModel(disc, model::AbstractElectrodeModel)

Discrete forward model of the electrode model `model` on the discretization `disc`.

Fields: `n` (system size; `n_u` for most models, `n_u + L` for the complete electrode model),
`n_u`, `n_σ`, `A` (system matrix, updated by [`system_matrix!`](@ref)), `A₀` (constant part of
its stored values or `nothing`), `tensor` ([`ConductivityTensor`](@ref)), `P` (injection,
`n × n_inject`), `Q` (measurement, `n_measure × n`), `nullspace`, `grounding`
(the functional `w` with `wᵀx = 0` for the current-driven solution), `measure_weights`
(weights of the mean removed from measured voltages),
`dirichlet_dofs`, `free_dofs`, `E` (Dirichlet expansion), `C` (current representation),
`angles` (angular positions of the injection sites, used by [`trigonometric_patterns`](@ref)).

See [`forward_neumann`](@ref) and [`forward_dirichlet`](@ref).
"""
struct ForwardModel{M <: AbstractElectrodeModel, CT, CF} <: AbstractForwardModel
    model::M
    n::Int
    n_u::Int
    n_σ::Int
    A::SparseMatrixCSC{Float64, Int}
    A₀::Union{Nothing, Vector{Float64}}
    tensor::CT
    P::SparseMatrixCSC{Float64, Int}
    Q::SparseMatrixCSC{Float64, Int}
    nullspace::Vector{Float64}
    grounding::Vector{Float64}
    dirichlet_dofs::Vector{Int}
    free_dofs::Vector{Int}
    E::SparseMatrixCSC{Float64, Int}
    C::SparseMatrixCSC{Float64, Int}
    C_fac::CF
    A_ff::SparseMatrixCSC{Float64, Int}
    ff_map::Vector{Int}
    angles::Vector{Float64}
    kohn_vogelius_consistent::Bool
    measure_weights::Vector{Float64}
end

function _forward_model(model, n_u, n_σ, A, A₀, tensor, P, Q, nullspace, grounding, B, E, angles, kv;
                        measure_weights = ones(size(Q, 1)))
    n = size(A, 1)
    free = setdiff(1:n, B)
    Aidx = SparseMatrixCSC(n, n, A.colptr, A.rowval, collect(1.0:nnz(A)))
    Aidx_ff = Aidx[free, free]
    ff_map = round.(Int, nonzeros(Aidx_ff))
    A_ff = SparseMatrixCSC(Aidx_ff.m, Aidx_ff.n, Aidx_ff.colptr, Aidx_ff.rowval, zeros(nnz(Aidx_ff)))
    C = sparse(E' * P[B, :])
    C_fac = isapprox(C, I) ? nothing : cholesky(Symmetric(C))
    return ForwardModel(model, n, n_u, n_σ, A, A₀, tensor, sparse(P), sparse(Q), nullspace, grounding,
                        collect(B), free, sparse(E), C, C_fac, A_ff, ff_map, angles, kv,
                        collect(Float64, measure_weights))
end

"""
    n_inject(fm), n_measure(fm), n_control(fm)

Number of injection sites (columns of `P`), measurements (rows of `Q`) and Dirichlet controls
(columns of `E`; equals `n_inject`).
"""
n_inject(fm::ForwardModel) = size(fm.P, 2)
n_measure(fm::ForwardModel) = size(fm.Q, 1)
n_control(fm::ForwardModel) = size(fm.E, 2)

"""
    system_matrix!(fm, σ)

Update `fm.A = A₀ + L(σ)` (and its Dirichlet block) for the conductivity coefficients `σ`;
returns `fm.A`. Non-positive coefficients throw an [`InfeasibleConductivityError`](@ref), which
the optimizers treat as a rejected trial point.
"""
function system_matrix!(fm::ForwardModel, σ::AbstractVector)
    length(σ) == fm.n_σ || throw(DimensionMismatch("σ must have $(fm.n_σ) entries"))
    all(>(0), σ) || throw(InfeasibleConductivityError("the conductivity must be positive"))
    assemble_weighted_stiffness!(fm.A, fm.tensor, σ; A₀ = fm.A₀)
    Af, A = nonzeros(fm.A_ff), nonzeros(fm.A)
    @inbounds for (i, p) in enumerate(fm.ff_map)
        Af[i] = A[p]
    end
    return fm.A
end

"""
    trigonometric_patterns(fm, K)

`n_inject × 2K` matrix of trigonometric patterns `cos(kθ)`, `sin(kθ)`, `k = 1..K`, at the
angular positions of the injection sites, shifted so that each pattern injects zero net current
(for the continuum model: `∫ g ds = 0`). Also usable as voltage patterns.
"""
function trigonometric_patterns(fm::ForwardModel, K::Integer)
    θ = fm.angles
    G = reduce(hcat, [f.(k .* θ) for k in 1:K for f in (cos, sin)])
    w = vec(sum(fm.P; dims = 1))          # net current of a unit pattern entry
    return G .- (w' * G) ./ sum(w)
end

# X[B, :] = E F, X[I, :] = A_II⁻¹ (-A_IB E F); AX, Xf, Rf are buffers (n × s, nf × s, nf × s)
function _dirichlet_solve!(X, st, fm::ForwardModel, F, AX, Xf, Rf)
    fill!(X, 0)
    mul!(view(X, fm.dirichlet_dofs, :), fm.E, F)
    mul!(AX, fm.A, X)
    Rf .= .-view(AX, fm.free_dofs, :)
    _solve!(Xf, st, Rf)
    view(X, fm.free_dofs, :) .= Xf
    return X
end

# currents Y = C⁻¹ Eᵀ (A X)_B
function _dirichlet_currents!(Y, fm::ForwardModel, X, AX)
    mul!(AX, fm.A, X)
    mul!(Y, fm.E', view(AX, fm.dirichlet_dofs, :))
    fm.C_fac === nothing || copyto!(Y, fm.C_fac \ Y)
    return Y
end

_init_neumann_solver(solver, fm::ForwardModel) = _init_solver(solver, fm.A, fm.nullspace, fm.grounding)
_init_dirichlet_solver(solver, fm::ForwardModel) =
    _init_solver(solver, fm.A_ff, zeros(size(fm.A_ff, 1), 0), zeros(size(fm.A_ff, 1), 0))

_like_input(Y, x::AbstractVector) = vec(Y)
_like_input(Y, ::AbstractMatrix) = Y

"""
    forward_neumann(fm, σ, currents; solver = DirectSolver()) -> (voltages, X)

Current-driven forward problem: solve `A(σ) X = P currents` (grounded) and measure
`voltages = Q X`. `currents` is a vector or an `n_inject × s` matrix of patterns; `X` holds
the full states (`n × s`).
"""
function forward_neumann(fm::ForwardModel, σ::AbstractVector, currents::AbstractVecOrMat;
                         solver::AbstractLinearSolver = DirectSolver())
    Cm = _as_matrix(currents)
    size(Cm, 1) == n_inject(fm) || throw(DimensionMismatch("currents need $(n_inject(fm)) rows"))
    system_matrix!(fm, σ)
    st = _init_neumann_solver(solver, fm)
    X = zeros(fm.n, size(Cm, 2))
    _solve!(X, st, fm.P * Cm)
    return _like_input(fm.Q * X, currents), X
end

"""
    forward_dirichlet(fm, σ, voltages; solver = DirectSolver()) -> (currents, X)

Voltage-driven forward problem: prescribe `voltages` (`n_control × s`) on the Dirichlet dofs
(boundary nodes, electrode nodes or CEM electrode voltages), solve for the rest and return the
currents in the representation of the injection (so that it inverts [`forward_neumann`](@ref)
for consistent data).
"""
function forward_dirichlet(fm::ForwardModel, σ::AbstractVector, voltages::AbstractVecOrMat;
                           solver::AbstractLinearSolver = DirectSolver())
    F = _as_matrix(voltages)
    size(F, 1) == n_control(fm) || throw(DimensionMismatch("voltages need $(n_control(fm)) rows"))
    system_matrix!(fm, σ)
    st = _init_dirichlet_solver(solver, fm)
    s, nf = size(F, 2), length(fm.free_dofs)
    X, AX = zeros(fm.n, s), zeros(fm.n, s)
    _dirichlet_solve!(X, st, fm, F, AX, zeros(nf, s), zeros(nf, s))
    Y = zeros(n_control(fm), s)
    _dirichlet_currents!(Y, fm, X, AX)
    return _like_input(Y, voltages), X
end
