# Electrode models on a Ferrite discretization. Each model maps a vector of injected currents to
# a load vector (P) and the state to a vector of measured voltages (Q); see ForwardModel.jl.
#
#   continuum:  every boundary dof is an electrode. Currents are boundary current densities
#               (coefficients at the boundary dofs), P g = ∫_Γ g φᵢ; voltages are the boundary
#               values of u.
#   point:      current enters at single boundary nodes, voltages are read at single nodes.
#   gap:        current I_ℓ is spread uniformly over electrode e_ℓ, P = ∫_{e_ℓ} φᵢ / |e_ℓ|;
#               the voltage is the mean potential on the electrode (Q = Pᵀ for the same
#               electrodes). Injection and measurement electrodes may differ.
#   complete (CEM): the electrode voltages U_ℓ are extra unknowns, contact impedances z_ℓ;
#               the voltage is measured on the electrodes, including current-carrying ones.
#
# Electrodes are vectors of boundary facets; `angular_electrodes` builds them for a ring of
# electrodes around a centre point (discs, and squares from pixel images).

"""
    ContinuumModel()

Continuum model: every boundary dof acts as an electrode. Current patterns are boundary current
densities (one value per boundary dof, `disc.boundary_dofs` order), voltages are the boundary
values of the potential.
"""
struct ContinuumModel <: AbstractElectrodeModel end

"""
    PointElectrodeModel(points; measure = points)

Point electrodes: current `I_ℓ` enters at the boundary dof closest to `points[ℓ]`, voltages are
read at the boundary dofs closest to `measure`. Injection and measurement points may differ.
"""
struct PointElectrodeModel{V} <: AbstractElectrodeModel
    inject::Vector{V}
    measure::Vector{V}
end
PointElectrodeModel(points; measure = points) = PointElectrodeModel(collect(points), collect(measure))

"""
    GapModel(electrodes; measure = electrodes)

Gap model: current `I_ℓ` is uniformly distributed over electrode `electrodes[ℓ]` (a vector of
boundary `FacetIndex`), zero in the gaps. The voltage of a measurement electrode is the mean
potential on it. Injection and measurement electrodes may differ (`measure`).
"""
struct GapModel <: AbstractElectrodeModel
    inject::Vector{Vector{FacetIndex}}
    measure::Vector{Vector{FacetIndex}}
end
GapModel(electrodes; measure = electrodes) = GapModel(collect.(electrodes), collect.(measure))

"""
    CompleteElectrodeModel(electrodes, z; measure = 1:length(electrodes))

Complete electrode model with contact impedances `z` (scalar or one per electrode). The electrode
voltages `U` are unknowns of the forward problem (grounded by `Σ U_ℓ = 0`); `measure` selects the
electrodes whose voltages are measured (e.g. only electrodes that carry no current).
"""
struct CompleteElectrodeModel <: AbstractElectrodeModel
    electrodes::Vector{Vector{FacetIndex}}
    z::Vector{Float64}
    measure::Vector{Int}
end
function CompleteElectrodeModel(electrodes, z; measure = 1:length(electrodes))
    L = length(electrodes)
    zv = z isa Number ? fill(Float64(z), L) : collect(Float64, z)
    length(zv) == L || throw(DimensionMismatch("need one contact impedance per electrode"))
    all(>(0), zv) || throw(ArgumentError("contact impedances must be positive"))
    all(in(1:L), measure) || throw(ArgumentError("measure must index electrodes 1:$L"))
    return CompleteElectrodeModel(collect.(electrodes), zv, collect(measure))
end

"""
    angular_electrodes(disc, L; coverage = 0.5, offset = 0.0, center = nothing)

`L` electrodes around `center` (default: centroid of the boundary curve), centred at
the angles `offset + 2π(ℓ-1)/L`. Electrode ℓ is the set of boundary facets whose midpoint lies
within `± coverage π / L` of its angle, so `coverage` is the fraction of the boundary covered.
"""
function angular_electrodes(disc::FerriteDiscretization, L::Integer; coverage::Real = 0.5,
                            offset::Real = 0.0, center = nothing)
    0 < coverage < 1 || throw(ArgumentError("coverage must be in (0, 1)"))
    mids = [_facet_midpoint(disc.grid, f) for f in disc.boundary_facets]
    c = center === nothing ? _centroid(disc, disc.boundary_facets) : Vec{2}(Tuple(center))
    θ = [atan(m[2] - c[2], m[1] - c[1]) for m in mids]
    half = coverage * π / L
    return map(1:L) do ℓ
        θℓ = offset + 2π * (ℓ - 1) / L
        [f for (f, t) in zip(disc.boundary_facets, θ) if abs(mod(t - θℓ + π, 2π) - π) < half]
    end
end

"""
    electrode_length(disc, facets)

Length (area in 3D) of an electrode given by its boundary facets.
"""
electrode_length(disc::FerriteDiscretization, facets) = sum(f -> _facet_measure(disc, f), facets; init = 0.0)

# u dof coordinates (x, y) of the dofs `dofs`
function _dof_coordinates(disc::FerriteDiscretization, dofs)
    x = interpolate_function(disc, p -> p[1]; field = :u)
    y = interpolate_function(disc, p -> p[2]; field = :u)
    return x[dofs], y[dofs]
end

# length-weighted centroid of a set of (straight) boundary facets: mesh independent, unlike
# averages over facets or nodes, which shift when the boundary is refined
function _centroid(disc::FerriteDiscretization, facets)
    lens = [_facet_measure(disc, f) for f in facets]
    return sum(l * _facet_midpoint(disc.grid, f) for (l, f) in zip(lens, facets)) / sum(lens)
end

# angles around the centroid of the boundary curve
function _angles(disc::FerriteDiscretization, xs, ys)
    c = _centroid(disc, disc.boundary_facets)
    return atan.(ys .- c[2], xs .- c[1])
end

function _electrode_angle(disc::FerriteDiscretization, facets)
    m = _centroid(disc, facets)
    return _angles(disc, [m[1]], [m[2]])[1]
end

# boundary load vector wᵢ = ∫_Γ φᵢ ds of the u space: wᵀu = ∫_Γ u ds
_boundary_weights(disc::FerriteDiscretization) =
    _condense(disc, assemble_boundary_load!(zeros(ndofs(disc.dh_u)), disc.dh_u, disc.fv_u, disc.boundary_facets))

function _grounding(disc::FerriteDiscretization, kind::Symbol)
    kind === :integral && return _boundary_weights(disc)
    kind === :nodal && return boundary_grounding(ndofs_u(disc), disc.boundary_dofs)
    throw(ArgumentError("grounding must be :integral or :nodal, got :$kind"))
end

# nᵤ × L matrix with columns ∫_{e_ℓ} φᵢ ds / |e_ℓ|
function _electrode_averages(disc::FerriteDiscretization, electrodes)
    nu = ndofs_u(disc)
    cols = map(electrodes) do e
        isempty(e) && throw(ArgumentError("empty electrode"))
        f = _condense(disc, assemble_boundary_load!(zeros(ndofs(disc.dh_u)), disc.dh_u, disc.fv_u, e))
        sparse(f ./ sum(f))
    end
    return reduce(hcat, cols)
end

function _nearest_boundary_dofs(disc::FerriteDiscretization, points)
    bx, by = _dof_coordinates(disc, disc.boundary_dofs)
    return [disc.boundary_dofs[argmin((bx .- p[1]) .^ 2 .+ (by .- p[2]) .^ 2)] for p in points]
end

"""
    ForwardModel(disc, model; grounding = :integral)

Discrete forward model of an electrode model on a Ferrite discretization (see
[`ForwardModel`](@ref)). `grounding` fixes the additive constant of the current-driven
potential for the continuum, point and gap models:

- `:integral` (default): zero boundary mean, `∫_Γ u ds = 0`. It does not depend on the mesh,
  so voltages computed on different meshes (e.g. adaptively refined ones) are comparable.
- `:nodal`: the boundary nodal values sum to zero. Equal to `:integral` up to a constant factor
  on meshes with uniformly spaced boundary nodes.

The complete electrode model always grounds the electrode voltages, `Σ U_ℓ = 0`.
"""
function ForwardModel(disc::FerriteDiscretization, model::ContinuumModel; grounding::Symbol = :integral)
    nu, bd = ndofs_u(disc), disc.boundary_dofs
    nb = length(bd)
    MΓ = _condense(disc, assemble_boundary_mass(disc.dh_u, disc.fv_u, disc.boundary_facets))
    P = MΓ[:, bd]
    Q = sparse(1:nb, bd, ones(nb), nb, nu)
    ct = ConductivityTensor(disc)
    θ = _angles(disc, _dof_coordinates(disc, bd)...)
    return _forward_model(model, nu, ndofs_σ(disc), copy(ct.pattern), nothing, ct, P, Q, ones(nu),
                          _grounding(disc, grounding), bd, sparse(1.0I, nb, nb), θ, true;
                          measure_weights = _boundary_weights(disc)[bd])
end

function ForwardModel(disc::FerriteDiscretization, model::PointElectrodeModel; grounding::Symbol = :integral)
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

function ForwardModel(disc::FerriteDiscretization, model::GapModel; grounding::Symbol = :integral)
    nu = ndofs_u(disc)
    P = _electrode_averages(disc, model.inject)
    Q = sparse(_electrode_averages(disc, model.measure)')
    # Dirichlet (voltage-driven) version: u = U_ℓ on all dofs of electrode ℓ (shunt model)
    edofs = [disc.full_to_free[_facet_dofs(disc.dh_u, disc.ip_u, e)] for e in model.inject]
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
function ForwardModel(disc::FerriteDiscretization, model::CompleteElectrodeModel; grounding::Symbol = :integral)
    nu, L = ndofs_u(disc), length(model.electrodes)
    n = nu + L
    Cuu = spzeros(nu, nu)
    D = spzeros(nu, L)
    lens = zeros(L)
    for (ℓ, e) in enumerate(model.electrodes)
        isempty(e) && throw(ArgumentError("electrode $ℓ is empty"))
        z = model.z[ℓ]
        Cuu += _condense(disc, assemble_boundary_mass(disc.dh_u, disc.fv_u, e)) ./ z
        d = _condense(disc, assemble_boundary_load!(zeros(ndofs(disc.dh_u)), disc.dh_u, disc.fv_u, e))
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
