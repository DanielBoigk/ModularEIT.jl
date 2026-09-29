# Fast solver for the constant-coefficient Neumann problem on rotationally symmetric disk meshes
# (wiki: Fast Solvers on Disk Domains).
#
# Nodes: an optional centre node, then nr rings of nθ nodes, ordered ring by ring (index
# 1 + c + (k-1) nθ + j for ring k = 1..nr, angle index j = 0..nθ-1, c = 1 with a centre node).
# If the mesh is invariant under the rotation by 2π/nθ, the stiffness matrix is block circulant
# in j: (K U)_j = Σ_δ K_δ U_{j+δ} (+ centre coupling), with nr × nr blocks K_δ. The DFT in j,
# Û_m = Σ_j U_j e^{-2πi jm/nθ}, decouples the angular modes:
#
#     K̂_m Û_m = b̂_m,   K̂_m = Σ_δ K_δ e^{2πi δ m/nθ}      (m = 1 … nθ/2; tridiagonal Hermitian)
#
# Only mode 0 couples to the centre node (whose row is the same for every node of the first
# ring) and contains the constants. It is solved bordered with the grounding Σᵢ uᵢ = 0:
#
#     [ K_cc   κᵀ     1  ] [u_c]   [b_c]
#     [ nθ κ   K̂_0   nθ 1 ] [Û_0] = [b̂_0]        (κ_k = K_{c,(k,j)} for all j)
#     [ 1      1ᵀ     0  ] [ c ]   [ 0 ]
#
# which gives the Moore–Penrose pseudo-inverse (mean-zero solution). The blocks are read off the
# assembled stiffness matrix, and the rotational invariance is verified entry by entry.

"""
    PolarStructure

Rotationally symmetric node structure of a disk mesh (`nr` rings of `nθ` nodes, optionally a
centre node) with the permutation `perm` from structure order to the u dofs, and the mode
factorisations of the FFT solver. Built by `polar_structure(disc)`.
"""
struct PolarStructure
    nr::Int
    nθ::Int
    center::Bool
    perm::Vector{Int}
    Kdiag::Vector{Float64}
    mode0::Any                 # factorisation of the bordered mode-0 system
    modes::Vector{Any}         # factorisations of K̂_m, m = 1 … nθ ÷ 2
end

function PolarStructure(nr::Integer, nθ::Integer, center::Bool, perm::AbstractVector{<:Integer}, K::SparseMatrixCSC;
                        rtol::Real = 1e-10)
    c = Int(center)
    n = c + nr * nθ
    size(K) == (n, n) || throw(DimensionMismatch("stiffness matrix must be $n × $n"))
    fail(msg) = throw(ArgumentError("the mesh is not rotationally symmetric: $msg"))
    ring(i) = (i - c - 1) ÷ nθ + 1
    ang(i) = (i - c - 1) % nθ
    tol = rtol * maximum(abs, nonzeros(K))
    blocks = Dict{Int, Matrix{Float64}}()                     # δ → K_δ
    counts = Dict{NTuple{3, Int}, Int}()
    κ = zeros(nr)
    κcount = zeros(Int, nr)
    Kcc = 0.0
    rows, vals = rowvals(K), nonzeros(K)
    for q in 1:n, idx in nzrange(K, q)
        p, v = rows[idx], vals[idx]
        if center && p == 1 && q == 1
            Kcc = v
        elseif center && (p == 1 || q == 1)
            p == 1 || continue                                # use row 1 (K symmetric)
            k = ring(q)
            if κcount[k] == 0
                κ[k] = v
            elseif abs(v - κ[k]) > tol
                fail("the centre couples differently to ring $k")
            end
            κcount[k] += 1
        else
            k, kk = ring(p), ring(q)
            δ = mod(ang(q) - ang(p), nθ)
            B = get!(() -> zeros(nr, nr), blocks, δ)
            key = (k, kk, δ)
            cnt = get(counts, key, 0)
            if cnt == 0
                B[k, kk] = v
            elseif abs(v - B[k, kk]) > tol
                fail("entry ($k, $kk) at angular offset $δ depends on the angle")
            end
            counts[key] = cnt + 1
        end
    end
    all(==(nθ), values(counts)) || fail("couplings are missing for some angles")
    center && any(k -> κcount[k] ∉ (0, nθ), 1:nr) && fail("the centre does not couple to whole rings")
    # mode matrices
    Khat(m) = sum(B .* cis(2π * δ * m / nθ) for (δ, B) in blocks)
    function factor(M)
        istridiagonal = all(abs(M[i, j]) <= tol for i in 1:nr, j in 1:nr if abs(i - j) > 1)
        return istridiagonal ? lu(Tridiagonal(diag(M, -1), diag(M), diag(M, 1))) : lu(M)
    end
    modes = Any[factor(Khat(m)) for m in 1:(nθ ÷ 2)]
    K0 = real(Khat(0))
    M0 = zeros(c + nr + 1, c + nr + 1)
    M0[c + 1:c + nr, c + 1:c + nr] .= K0
    M0[c + 1:c + nr, end] .= nθ
    M0[end, c + 1:c + nr] .= 1
    if center
        M0[1, 1] = Kcc
        M0[1, 2:nr + 1] .= κ
        M0[2:nr + 1, 1] .= nθ .* κ
        M0[1, end] = 1
        M0[end, 1] = 1
    end
    return PolarStructure(nr, nθ, center, collect(Int, perm), Vector(diag(K)), lu(M0), modes)
end

"""
    fast_neumann_solve(structure, B)

Moore–Penrose pseudo-inverse of the stiffness matrix of a [`PolarStructure`](@ref) (or of a
[`StructuredGrid`](@ref)) applied to `B` (vector or `n × s`, in structure order): FFT in the
angle, one tridiagonal solve per angular mode.
"""
function fast_neumann_solve(p::PolarStructure, B::AbstractVecOrMat)
    Bm = _as_matrix(B)
    s = size(Bm, 2)
    c, nr, nθ = Int(p.center), p.nr, p.nθ
    R = reshape(Bm[c + 1:end, :], nθ, nr, s)
    F = AbstractFFTs.rfft(R, 1)                              # (nθ ÷ 2 + 1) × nr × s
    Y = similar(F)
    uc = zeros(s)
    for t in 1:s
        rhs0 = [p.center ? [Bm[1, t]] : Float64[]; real.(F[1, :, t]); 0.0]
        sol = p.mode0 \ rhs0
        p.center && (uc[t] = sol[1])
        Y[1, :, t] .= sol[c + 1:c + nr]
        for m in 1:(nθ ÷ 2)
            Y[m + 1, :, t] .= p.modes[m] \ F[m + 1, :, t]
        end
    end
    U = AbstractFFTs.irfft(Y, nθ, 1)
    out = p.center ? [uc'; reshape(U, nθ * nr, s)] : reshape(U, nθ * nr, s)
    return B isa AbstractVector ? vec(out) : out
end

fast_neumann_solve(g::StructuredGrid, B::AbstractVecOrMat) = dct_neumann_solve(g, B)

# interface of the fast solvers used by the system preconditioners
_fast_n(p::PolarStructure) = length(p.perm)
_fast_n(g::StructuredGrid) = g.nx * g.ny
_fast_perm(x::Union{PolarStructure, StructuredGrid}) = x.perm
_fast_kdiag(p::PolarStructure) = p.Kdiag
function _fast_kdiag(g::StructuredGrid)                    # Q1: K = Kx ⊗ My + Mx ⊗ Ky
    kx = [(i == 1 || i == g.nx ? 1 : 2) / g.hx for i in 1:g.nx]
    mx = [(i == 1 || i == g.nx ? 2 : 4) * g.hx / 6 for i in 1:g.nx]
    ky = [(j == 1 || j == g.ny ? 1 : 2) / g.hy for j in 1:g.ny]
    my = [(j == 1 || j == g.ny ? 2 : 4) * g.hy / 6 for j in 1:g.ny]
    return vec(kx .* my' .+ mx .* ky')
end
