# Electrodes and forward models for every back end. A back end provides the electrode
# primitives below for its discretization type; electrode placement, the grounding and the forward
# models of the four electrode models are built from them here.
#
# Electrode primitives (back end contract, internal):
#   _boundary_facets(disc)          boundary facets (the electrode representation's element type)
#   _boundary_dofs(disc)            free u dofs on the boundary
#   _boundary_mass(disc, facets)    nᵤ × nᵤ matrix ∫_facets φᵢ φⱼ ds (free dofs)
#   _boundary_load(disc, facets)    nᵤ vector ∫_facets φᵢ ds (free dofs)
#   _facet_free_dofs(disc, facets)  free u dofs on the facets
#   _facet_measure(disc, f), _facet_midpoint(disc, f), _facet_vertices(disc, f) (2D: end points)
#   _u_pattern(disc)                sparsity pattern of the u space
#   _spatial_dim(disc)

using LinearAlgebra
using SparseArrays

function _boundary_facets end
function _boundary_dofs end
function _boundary_mass end
function _boundary_load end
function _facet_free_dofs end
function _facet_measure end
function _facet_midpoint end
function _facet_vertices end
function _u_pattern end
function _spatial_dim end

"""
    angular_electrodes(disc, L; coverage = 0.5, offset = 0.0, center = nothing, angles = nothing)

`L` electrodes around `center` (default: centroid of the boundary curve), centred at
the angles `offset + 2π(ℓ-1)/L`, or at the given `angles` (e.g. perturbed positions from
[`electrode_angles`](@ref)). Electrode ℓ is the set of boundary facets whose midpoint lies
within `± coverage π / L` of its angle, so `coverage` is the fraction of the boundary covered.
"""
function angular_electrodes(disc::AbstractDiscretization, L::Integer; coverage::Real = 0.5,
                            offset::Real = 0.0, center = nothing, angles = nothing)
    0 < coverage < 1 || throw(ArgumentError("coverage must be in (0, 1)"))
    angles === nothing || length(angles) == L || throw(DimensionMismatch("need $L electrode angles"))
    bf = _boundary_facets(disc)
    mids = [_facet_midpoint(disc, f) for f in bf]
    c = center === nothing ? _centroid(disc, bf) : (Float64(center[1]), Float64(center[2]))
    θ = [atan(m[2] - c[2], m[1] - c[1]) for m in mids]
    half = coverage * π / L
    return map(1:L) do ℓ
        θℓ = angles === nothing ? offset + 2π * (ℓ - 1) / L : angles[ℓ]
        [f for (f, t) in zip(bf, θ) if abs(mod(t - θℓ + π, 2π) - π) < half]
    end
end

"""
    electrode_length(disc, facets)

Length (area in 3D) of an electrode given by its boundary facets.
"""
electrode_length(disc::AbstractDiscretization, facets) = sum(f -> _facet_measure(disc, f), facets; init = 0.0)

# u dof coordinates (x, y) of the dofs `dofs`
function _dof_coordinates(disc::AbstractDiscretization, dofs)
    x = interpolate_function(disc, p -> p[1]; field = :u)
    y = interpolate_function(disc, p -> p[2]; field = :u)
    return x[dofs], y[dofs]
end

# length-weighted centroid of a set of (straight) boundary facets: mesh independent, unlike
# averages over facets or nodes, which shift when the boundary is refined
function _centroid(disc::AbstractDiscretization, facets)
    lens = [_facet_measure(disc, f) for f in facets]
    return sum(l * _facet_midpoint(disc, f) for (l, f) in zip(lens, facets)) / sum(lens)
end

# angles around the centroid of the boundary curve
function _angles(disc::AbstractDiscretization, xs, ys)
    c = _centroid(disc, _boundary_facets(disc))
    return atan.(ys .- c[2], xs .- c[1])
end

function _electrode_angle(disc::AbstractDiscretization, facets)
    m = _centroid(disc, facets)
    return _angles(disc, [m[1]], [m[2]])[1]
end

# boundary load vector wᵢ = ∫_Γ φᵢ ds of the u space: wᵀu = ∫_Γ u ds
_boundary_weights(disc::AbstractDiscretization) = _boundary_load(disc, _boundary_facets(disc))

function _grounding(disc::AbstractDiscretization, kind::Symbol)
    kind === :integral && return _boundary_weights(disc)
    kind === :nodal && return boundary_grounding(ndofs_u(disc), _boundary_dofs(disc))
    throw(ArgumentError("grounding must be :integral or :nodal, got :$kind"))
end

# nᵤ × L matrix with columns ∫_{e_ℓ} φᵢ ds / |e_ℓ|
function _electrode_averages(disc::AbstractDiscretization, electrodes)
    nu = ndofs_u(disc)
    cols = map(electrodes) do e
        isempty(e) && throw(ArgumentError("empty electrode"))
        f = _boundary_load(disc, e)
        sparse(f ./ sum(f))
    end
    return reduce(hcat, cols)
end

function _nearest_boundary_dofs(disc::AbstractDiscretization, points)
    bd = _boundary_dofs(disc)
    bx, by = _dof_coordinates(disc, bd)
    return [bd[argmin((bx .- p[1]) .^ 2 .+ (by .- p[2]) .^ 2)] for p in points]
end

"""
    ForwardModel(disc, model; grounding = :integral)

Discrete forward model of an electrode model on a discretization (see
[`ForwardModel`](@ref)); works for every back end through the electrode primitives. `grounding` fixes the additive constant of the current-driven
potential for the continuum, point and gap models:

- `:integral` (default): zero boundary mean, `∫_Γ u ds = 0`. It does not depend on the mesh,
  so voltages computed on different meshes (e.g. adaptively refined ones) are comparable.
- `:nodal`: the boundary nodal values sum to zero. Equal to `:integral` up to a constant factor
  on meshes with uniformly spaced boundary nodes.

The complete electrode model always grounds the electrode voltages, `Σ U_ℓ = 0`.
"""
function ForwardModel(disc::AbstractDiscretization, model::ContinuumModel; grounding::Symbol = :integral)
    nu, bd = ndofs_u(disc), _boundary_dofs(disc)
    nb = length(bd)
    MΓ = _boundary_mass(disc, _boundary_facets(disc))
    P = MΓ[:, bd]
    Q = sparse(1:nb, bd, ones(nb), nb, nu)
    ct = ConductivityTensor(disc)
    θ = _angles(disc, _dof_coordinates(disc, bd)...)
    return _forward_model(model, nu, ndofs_σ(disc), copy(ct.pattern), nothing, ct, P, Q, ones(nu),
                          _grounding(disc, grounding), bd, sparse(1.0I, nb, nb), θ, true;
                          measure_weights = _boundary_weights(disc)[bd])
end

function ForwardModel(disc::AbstractDiscretization, model::PointElectrodeModel; grounding::Symbol = :integral)
    nu = ndofs_u(disc)
    inj = _nearest_boundary_dofs(disc, model.inject)
    meas = _nearest_boundary_dofs(disc, model.measure)
    allunique(inj) || throw(ArgumentError("two injection points share the nearest boundary dof"))
    m, mm = length(inj), length(meas)
    P = sparse(inj, 1:m, ones(m), nu, m)
    Q = sparse(1:mm, meas, ones(mm), mm, nu)
    ct = ConductivityTensor(disc)
    θ = _angles(disc, _dof_coordinates(disc, inj)...)
    return _forward_model(model, nu, ndofs_σ(disc), copy(ct.pattern), nothing, ct, P, Q, ones(nu),
                          _grounding(disc, grounding), inj, sparse(1.0I, m, m), θ,
                          sort(inj) == sort(meas))
end

function ForwardModel(disc::AbstractDiscretization, model::GapModel; grounding::Symbol = :integral)
    nu = ndofs_u(disc)
    P = _electrode_averages(disc, model.inject)
    Q = sparse(_electrode_averages(disc, model.measure)')
    # Dirichlet (voltage-driven) version: u = U_ℓ on all dofs of electrode ℓ (shunt model)
    edofs = [_facet_free_dofs(disc, e) for e in model.inject]
    B = reduce(vcat, edofs)
    allunique(B) || throw(ArgumentError("injection electrodes share boundary dofs"))
    E = sparse(1:length(B), reduce(vcat, [fill(ℓ, length(d)) for (ℓ, d) in enumerate(edofs)]),
               ones(length(B)), length(B), length(edofs))
    ct = ConductivityTensor(disc)
    θ = [_electrode_angle(disc, e) for e in model.inject]
    return _forward_model(model, nu, ndofs_σ(disc), copy(ct.pattern), nothing, ct, P, Q, ones(nu),
                          _grounding(disc, grounding), B, E, θ, false)
end

# Complete electrode model, unknowns (u, U) ∈ ℝ^{nᵤ} × ℝ^L:
#
#   [ L(σ) + Σ_ℓ M_ℓ / z_ℓ      -d_ℓ / z_ℓ        ] [u]   [0]
#   [ -d_ℓᵀ / z_ℓ               |e_ℓ| / z_ℓ δ_ℓk  ] [U] = [I]
#
# with the electrode mass matrices M_ℓ = ∫_{e_ℓ} φᵢ φⱼ ds and d_ℓ = ∫_{e_ℓ} φᵢ ds.
function ForwardModel(disc::AbstractDiscretization, model::CompleteElectrodeModel; grounding::Symbol = :integral)
    nu, L = ndofs_u(disc), length(model.electrodes)
    n = nu + L
    Cuu = spzeros(nu, nu)
    D = spzeros(nu, L)
    lens = zeros(L)
    for (ℓ, e) in enumerate(model.electrodes)
        isempty(e) && throw(ArgumentError("electrode $ℓ is empty"))
        z = model.z[ℓ]
        Cuu += _boundary_mass(disc, e) ./ z
        d = _boundary_load(disc, e)
        D[:, ℓ] = sparse(d ./ z)
        lens[ℓ] = sum(d)
    end
    A0 = [Cuu -D; -D' spdiagm(lens ./ model.z)]
    # pattern: stiffness pattern of the u block ∪ pattern of A0
    Ipat, Jpat, _ = findnz(_u_pattern(disc))
    I0, J0, V0 = findnz(A0)
    pattern = sparse(vcat(Ipat, I0), vcat(Jpat, J0), ones(length(Ipat) + length(I0)), n, n)
    fill!(nonzeros(pattern), 0)
    A₀ = zeros(nnz(pattern))
    for (i, j, v) in zip(I0, J0, V0)
        A₀[_nz_index(pattern, i, j)] += v
    end
    ct = ConductivityTensor(disc; pattern)
    P = sparse(nu .+ (1:L), 1:L, ones(L), n, L)
    m = length(model.measure)
    Q = sparse(1:m, nu .+ model.measure, ones(m), m, n)
    grounding = [zeros(nu); ones(L)]
    θ = [_electrode_angle(disc, e) for e in model.electrodes]
    return _forward_model(model, nu, ndofs_σ(disc), copy(pattern), A₀, ct, P, Q, ones(n), grounding,
                          nu .+ (1:L), sparse(1.0I, L, L), θ, sort(model.measure) == 1:L)
end

"""
    transfer_electrodes(src, electrodes, dst)

The electrodes `electrodes` (boundary facets of the discretization `src`) on another 2D mesh
`dst` of the same domain: every boundary facet of `dst` belongs to the electrode of the nearest
boundary facet of `src` (distance of its midpoint to the facet segments). Exact for nested
meshes, e.g. to simulate data on a refined mesh with the same physical electrodes as the
reconstruction mesh (avoiding the inverse crime without changing the electrodes).
"""
function transfer_electrodes(src::AbstractDiscretization, electrodes, dst::AbstractDiscretization)
    _spatial_dim(src) == 2 || throw(ArgumentError("transfer_electrodes is implemented for 2D meshes"))
    label = Dict{eltype(_boundary_facets(src)), Int}()
    for (ℓ, e) in enumerate(electrodes), f in e
        label[f] = ℓ
    end
    segs = map(_boundary_facets(src)) do fi
        a, b = _facet_vertices(src, fi)
        (a, b, get(label, fi, 0))
    end
    bf = _boundary_facets(dst)
    out = [eltype(bf)[] for _ in electrodes]
    for fi in bf
        m = _facet_midpoint(dst, fi)
        _, k = findmin(s -> _segment_distance(m, s[1], s[2]), segs)
        ℓ = segs[k][3]
        ℓ > 0 && push!(out[ℓ], fi)
    end
    return out
end

function _segment_distance(p, a, b)
    d = b - a
    t = clamp(((p - a) ⋅ d) / (d ⋅ d), 0, 1)
    return norm(p - (a + t * d))
end
