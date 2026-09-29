# Fast-transform preconditioners for the systems of a ForwardModel: DCT on uniform rectangle grids,
# FFT in the angle on rotationally symmetric disk meshes (wiki: Fast Solvers on Rectangular
# Domains, Fast Solvers on Disk Domains). Both only provide a fast pseudo-inverse K⁺ of the
# constant-coefficient stiffness matrix; everything else here is shared.
#
# The system is either the full current-driven system A (size n) or the voltage-driven block
# A_ff on the free dofs. Its dofs split into u dofs (grid nodes) and extra dofs (the electrode
# voltages of the complete electrode model). With the constant part A₀ of the matrix (contact
# terms B on the u block, couplings, electrode block) and nodal conductivity estimates s,
#
#     A ≈ [ S^½ (K + S^-½ B S^-½) S^½    A_ue ]        S = σ̄ I (:constant) or diag(s) (:scaled),
#         [ A_ueᵀ                        A_ee ]
#
# restricted to the free u nodes (u = 0 on constrained nodes D). The u block (K + B_S) y = b with
# y_D = 0 is solved with the transform pseudo-inverse K⁺ (exact on the range of K, zero
# trapezoidal mean) and a capacitance correction for G = [E_B  E_D] (contact nodes: μ = B_S y;
# constrained nodes: Lagrange multipliers λ), writing y = K⁺(b - G ν) + c 1:
#
#     [ GᵀK⁺G + Ĉ   -Gᵀ1 ] [ν]   [ GᵀK⁺ b ]
#     [ -1ᵀG          0  ] [c] = [ -1ᵀ b  ],     Ĉ = blockdiag(B_S,r⁻¹, 0),
#
# the second row being the solvability condition 1ᵀ(b - Gν) = 0 for K. (A regularised K̃ =
# K + γ w wᵀ instead of the bordering cancels catastrophically: wᵀK̃⁻¹w = 1/γ ~ N⁴.) GᵀK⁺G is
# computed once (m transform solves); after conductivity updates only small matrices change. The extra dofs are eliminated with
# the L × L Schur complement (pseudo-inverse: it is singular for the current-driven CEM). For
# constant σ the preconditioner is the exact (pseudo-)inverse.
#
# Nodal conductivity estimates are read off the matrix: s_i = (A - A₀)_ii / K_ii.

"""
    AbstractFastPreconditioner

Preconditioner choice for [`BlockCGSolver`](@ref) that inverts the constant-conductivity system
with fast transforms: [`DCTPreconditioner`](@ref) (uniform rectangle grids),
[`PolarPreconditioner`](@ref) (rotationally symmetric disk meshes).
"""
abstract type AbstractFastPreconditioner end

"""
    DCTPreconditioner(disc; variant = :constant)

Choice of the DCT preconditioner for [`BlockCGSolver`](@ref) on a uniform rectangle grid of
bilinear elements (see `structured_grid`): the exact inverse of the system for a constant
conductivity, applied with fast cosine transforms (`O(n log n)` per application), for every
electrode model and for current- and voltage-driven problems. The number of CG iterations does
not grow with the mesh size, only with the conductivity contrast.

- `variant = :constant`: `σ̄ K` with the geometric mean `σ̄` of the nodal conductivities;
  iterations grow like `√(σmax/σmin)`.
- `variant = :scaled`: `S^½ K S^½` with the nodal conductivities `S` (Concus–Golub): for smooth
  conductivities the iterations then hardly depend on the contrast. Not for jumps: there the
  iteration count grows with mesh refinement.

The setup costs one transform solve per contact node (complete electrode model) and per
constrained node (voltage-driven problems), once per forward model.
"""
struct DCTPreconditioner{D} <: AbstractFastPreconditioner
    disc::D
    grid::StructuredGrid
    variant::Symbol
end
function DCTPreconditioner(disc::AbstractDiscretization; variant::Symbol = :constant)
    _check_variant(variant)
    return DCTPreconditioner(disc, structured_grid(disc), variant)
end

"""
    PolarPreconditioner(disc; variant = :constant)

Choice of the FFT preconditioner for [`BlockCGSolver`](@ref) on a rotationally symmetric disk
mesh of linear triangles, e.g. [`polar_grid`](@ref): the constant-conductivity system is
inverted with an FFT in the angle and one tridiagonal solve per angular mode along the radius
(`O(n log n)`), for every electrode model and for current- and voltage-driven problems. The
radial spacing is arbitrary (e.g. graded towards the boundary). Variants and costs as for
[`DCTPreconditioner`](@ref).

On a conformally mapped mesh ([`conformal_grid`](@ref)) pass the disk mesh as `reference`: the
stiffness matrix of the disk then preconditions the one of the mapped domain; since the map is
conformal, both are spectrally equivalent with constants close to 1.
"""
struct PolarPreconditioner{D} <: AbstractFastPreconditioner
    disc::D
    grid::PolarStructure
    variant::Symbol
end
function PolarPreconditioner(disc::AbstractDiscretization; variant::Symbol = :constant, reference = nothing)
    _check_variant(variant)
    ps = reference === nothing ? polar_structure(disc) : _reference_structure(disc, reference)
    return PolarPreconditioner(disc, ps, variant)
end

_check_variant(v) = v in (:constant, :scaled) || throw(ArgumentError("variant must be :constant or :scaled, got :$v"))

"""
    dct_preconditioner(disc, fm; system = :neumann, variant = :constant)

DCT preconditioner for the current-driven system `fm.A` (`system = :neumann`) or the
voltage-driven block `fm.A_ff` (`:dirichlet`) of a forward model, for use with [`pbcg!`](@ref)
(`M = …`). Call `update_preconditioner!(M, A)` after every change of the matrix values.
"""
dct_preconditioner(disc::AbstractDiscretization, fm::ForwardModel; system::Symbol = :neumann,
                   variant::Symbol = :constant) =
    _FastSystem(DCTPreconditioner(disc; variant), fm, system)

"""
    polar_preconditioner(disc, fm; system = :neumann, variant = :constant)

FFT preconditioner (see [`PolarPreconditioner`](@ref)) for the current-driven system `fm.A` or
the voltage-driven block `fm.A_ff` of a forward model on a disk mesh, for [`pbcg!`](@ref); call
[`update_preconditioner!`](@ref) after every change of the matrix values.
"""
polar_preconditioner(disc::AbstractDiscretization, fm::ForwardModel; system::Symbol = :neumann,
                     variant::Symbol = :constant, reference = nothing) =
    _FastSystem(PolarPreconditioner(disc; variant, reference), fm, system)

mutable struct _FastSystem{S}
    grid::S                    # StructuredGrid or PolarStructure
    variant::Symbol
    sys::Vector{Int}           # system index → dof of the forward model
    su::Vector{Int}            # system indices of u dofs
    su_lex::Vector{Int}        # their positions in the fast solver's node order
    se::Vector{Int}            # system indices of extra dofs (CEM electrode voltages)
    A0::SparseMatrixCSC{Float64, Int}   # constant part of the system matrix
    Kdiag::Vector{Float64}     # K_ii of the system u dofs
    b_lex::Vector{Int}         # contact nodes (lexicographic) and their system u positions
    b_su::Vector{Int}
    Br::Matrix{Float64}        # contact block
    d_lex::Vector{Int}         # constrained u nodes (lexicographic)
    hasG::Bool
    GtKG::Matrix{Float64}      # Gᵀ K⁺ G
    Aue::Matrix{Float64}       # u-extra coupling (constant)
    Aee::Matrix{Float64}
    # updated with the matrix
    sqs::Vector{Float64}       # √s on the grid (1 where unknown)
    cap::Any                   # factorisation of the capacitance matrix
    HinvAue::Matrix{Float64}
    Spinv::Matrix{Float64}
end

function _FastSystem(spec::AbstractFastPreconditioner, fm::ForwardModel, system::Symbol)
    system in (:neumann, :dirichlet) || throw(ArgumentError("system must be :neumann or :dirichlet"))
    g = spec.grid
    nu = fm.n_u
    length(_fast_perm(g)) == nu || throw(DimensionMismatch("the forward model does not belong to this discretization"))
    sys = system === :neumann ? collect(1:fm.n) : fm.free_dofs
    lex_of_dof = invperm(_fast_perm(g))
    su = findall(<=(nu), sys)
    se = findall(>(nu), sys)
    su_lex = lex_of_dof[sys[su]]
    A0full = fm.A₀ === nothing ? spzeros(fm.n, fm.n) :
             SparseMatrixCSC(fm.n, fm.n, fm.A.colptr, fm.A.rowval, copy(fm.A₀))
    A0 = A0full[sys, sys]
    dropzeros!(A0)
    Buu = A0[su, su]
    b_su = findall(i -> nnz(Buu[:, i]) > 0, 1:length(su))
    b_lex = su_lex[b_su]
    Br = Matrix(Buu[b_su, b_su])
    freeu = falses(nu)
    freeu[sys[su]] .= true
    d_lex = lex_of_dof[findall(!, freeu)]
    hasG = !isempty(b_lex) || !isempty(d_lex)
    Kdiag = _fast_kdiag(g)[su_lex]
    st = _FastSystem(g, spec.variant, sys, su, su_lex, se, A0, Kdiag, b_lex, b_su, Br, d_lex, hasG,
                    zeros(0, 0), Matrix(A0[su, se]), Matrix(A0[se, se]), ones(_fast_n(g)), nothing,
                    zeros(length(su), length(se)), zeros(length(se), length(se)))
    hasG && (st.GtKG = _capacitance_base(st))
    return st
end

_ncap(st::_FastSystem) = length(st.b_lex) + length(st.d_lex)

# G ν for a block ν (m × k) → grid vectors (nx ny × k); G = [E_B  E_D]
function _G_mul(st::_FastSystem, ν::AbstractMatrix)
    r = length(st.b_lex)
    T = zeros(_fast_n(st.grid), size(ν, 2))
    T[st.b_lex, :] .+= ν[1:r, :]
    T[st.d_lex, :] .+= ν[(r + 1):end, :]
    return T
end
_Gt_mul(st::_FastSystem, Y::AbstractMatrix) = [Y[st.b_lex, :]; Y[st.d_lex, :]]

function _capacitance_base(st::_FastSystem; chunk::Int = 64)
    m = _ncap(st)
    GtKG = zeros(m, m)
    for c0 in 1:chunk:m
        cols = c0:min(c0 + chunk - 1, m)
        E = zeros(m, length(cols))
        for (k, c) in enumerate(cols)
            E[c, k] = 1
        end
        GtKG[:, cols] = _Gt_mul(st, fast_neumann_solve(st.grid, _G_mul(st, E)))
    end
    return (GtKG + GtKG') / 2
end

"""
    update_preconditioner!(M, A)

Update a fast-transform preconditioner (from [`dct_preconditioner`](@ref) or
[`polar_preconditioner`](@ref)) to new values of its system matrix `A` (same sparsity pattern).
"""
function update_preconditioner!(st::_FastSystem, A::AbstractMatrix)
    size(A, 1) == length(st.sys) || throw(DimensionMismatch("matrix does not match the system"))
    sdiag = [(A[i, i] - st.A0[i, i]) for i in st.su] ./ st.Kdiag
    any(<=(0), sdiag) && throw(InfeasibleConductivityError("nonpositive nodal conductivity estimate"))
    fill!(st.sqs, 1.0)
    if st.variant === :constant
        σ̄ = exp(sum(log, sdiag) / length(sdiag))
        st.sqs .= sqrt(σ̄)
    else
        st.sqs[st.su_lex] .= sqrt.(sdiag)
    end
    if st.hasG
        r, m = length(st.b_lex), _ncap(st)
        Cap = zeros(m + 1, m + 1)
        Cap[1:m, 1:m] .= st.GtKG
        if r > 0
            sb = st.sqs[st.b_lex]
            Cap[1:r, 1:r] .+= inv(Symmetric(st.Br ./ (sb .* sb')))   # (S^-½ B S^-½)⁻¹ on contact nodes
        end
        Cap[1:m, m + 1] .= -1                                       # -Gᵀ1 (G has unit columns)
        Cap[m + 1, 1:m] .= -1
        st.cap = lu(Cap)
    end
    if !isempty(st.se)
        st.HinvAue = _apply_H(st, st.Aue)
        Sc = st.Aee - st.Aue' * st.HinvAue
        st.Spinv = pinv(Symmetric((Sc + Sc') / 2))
    end
    return st
end

# approximate inverse of the u block on the system u dofs (block of vectors)
function _apply_H(st::_FastSystem, R::AbstractMatrix)
    g = st.grid
    b = zeros(_fast_n(g), size(R, 2))
    b[st.su_lex, :] .= R ./ st.sqs[st.su_lex]
    y = fast_neumann_solve(g, b)
    if st.hasG
        m = _ncap(st)
        sol = st.cap \ [_Gt_mul(st, y); -sum(b; dims = 1)]
        y .-= fast_neumann_solve(g, _G_mul(st, sol[1:m, :]))
        y .+= sol[m + 1:m + 1, :]                                  # constant mode c
    end
    return y[st.su_lex, :] ./ st.sqs[st.su_lex]
end

function apply_preconditioner!(Z, st::_FastSystem, R)
    Rh = Matrix{Float64}(_as_matrix(R))
    Zh = zeros(size(Rh))
    U1 = _apply_H(st, Rh[st.su, :])
    if isempty(st.se)
        Zh[st.su, :] .= U1
    else
        Ue = st.Spinv * (Rh[st.se, :] .- st.Aue' * U1)
        Zh[st.su, :] .= U1 .- st.HinvAue * Ue
        Zh[st.se, :] .= Ue
    end
    copyto!(Z, Zh)
    return Z
end

# BlockCGSolver with a fast-transform preconditioner: the forward model provides the structure
function _init_neumann_solver(s::BlockCGSolver{<:AbstractFastPreconditioner}, fm::ForwardModel)
    st = _init_solver(s, fm.A, fm.nullspace, fm.grounding)
    st.pcstate = _FastSystem(s.preconditioner, fm, :neumann)
    return st
end
function _init_dirichlet_solver(s::BlockCGSolver{<:AbstractFastPreconditioner}, fm::ForwardModel)
    k = size(fm.A_ff, 1)
    st = _init_solver(s, fm.A_ff, zeros(k, 0), zeros(k, 0))
    st.pcstate = _FastSystem(s.preconditioner, fm, :dirichlet)
    return st
end
