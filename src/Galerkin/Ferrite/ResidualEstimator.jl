# Residual-based a posteriori error indicator for the conductivity equation ∇⋅(σ∇u) = 0:
#
#   η_K² = Σₛ ( h_K² ‖∇⋅(σ∇uₛ)‖²_K + ½ Σ_{F ⊂ ∂K interior} h_F ‖[σ ∂ₙuₛ]‖²_F
#                                   +   Σ_{F ⊂ ∂K ∩ ∂Ω}   h_F ‖gₛ - σ ∂ₙuₛ‖²_F ),
#
# with the prescribed boundary current density gₛ of the electrode model. Interior facets are
# the ones listed in `disc.interior_facets`; the neighbour's current density is evaluated at the
# facet quadrature points through a local-coordinate search in the neighbour cell, so conforming
# facets and the fine pieces of coarse–fine (hanging) facets are treated alike.

# prescribed boundary current densities
abstract type _BoundaryData end
struct _NoBoundary <: _BoundaryData end                          # all boundary facets Dirichlet
struct _ZeroFlux <: _BoundaryData                                # g = 0, except skipped facets
    skip::Set{FacetIndex}
end
struct _Density <: _BoundaryData                                 # g in the u space (full dofs)
    G::Matrix{Float64}
end
struct _ElectrodeCurrents <: _BoundaryData                       # g = I_ℓ/|e_ℓ| on electrode ℓ
    facet_el::Dict{FacetIndex, Int}
    lens::Vector{Float64}
    I::Matrix{Float64}
end
struct _Robin <: _BoundaryData                                   # CEM: g = (U_ℓ - u)/z_ℓ on e_ℓ
    facet_el::Dict{FacetIndex, Int}
    z::Vector{Float64}
    U::Matrix{Float64}
end

_facet_map(electrodes) = Dict(fi => ℓ for (ℓ, e) in enumerate(electrodes) for fi in e)

# (include facet?, g) at facet quadrature point q for pattern s; ue, ge: cell coefficients
_bdata(::_NoBoundary, fi, s, fv, q, ue, ge) = (false, 0.0)
_bdata(b::_ZeroFlux, fi, s, fv, q, ue, ge) = (!(fi in b.skip), 0.0)
_bdata(b::_Density, fi, s, fv, q, ue, ge) = (true, function_value(fv, q, ge))
function _bdata(b::_ElectrodeCurrents, fi, s, fv, q, ue, ge)
    ℓ = get(b.facet_el, fi, 0)
    return (true, ℓ == 0 ? 0.0 : b.I[ℓ, s] / b.lens[ℓ])
end
function _bdata(b::_Robin, fi, s, fv, q, ue, ge)
    ℓ = get(b.facet_el, fi, 0)
    return (true, ℓ == 0 ? 0.0 : (b.U[ℓ, s] - function_value(fv, q, ue)) / b.z[ℓ])
end

function _needs_currents(model, mode)
    mode === :neumann && (model isa ContinuumModel || model isa GapModel)
end

function _primal_boundary(disc::FerriteDiscretization, fm::ForwardModel, X, currents, mode::Symbol)
    model, nu = fm.model, ndofs_u(disc)
    if _needs_currents(model, mode) && currents === nothing
        throw(ArgumentError("the boundary residual of $(nameof(typeof(model))) needs the injected currents"))
    end
    if model isa ContinuumModel
        mode === :dirichlet && return _NoBoundary()
        Cm = _as_matrix(currents)
        G = zeros(nu, size(Cm, 2))
        G[disc.boundary_dofs, :] .= Cm
        return _Density(Matrix(_lift(disc, G)))
    elseif model isa PointElectrodeModel
        return _ZeroFlux(Set{FacetIndex}())       # point loads are singular: not estimated
    elseif model isa GapModel
        mode === :dirichlet && return _ZeroFlux(Set(fi for e in model.inject for fi in e))
        lens = [electrode_length(disc, e) for e in model.inject]
        return _ElectrodeCurrents(_facet_map(model.inject), lens, Matrix{Float64}(_as_matrix(currents)))
    elseif model isa CompleteElectrodeModel
        return _Robin(_facet_map(model.electrodes), model.z, Matrix(X[nu+1:end, :]))
    end
    throw(ArgumentError("no boundary residual for $(typeof(model))"))
end

# boundary data of the measurement duals A zₘ = Qᵀ Π eₘ (current-driven by the measurement
# functional; Π removes the mean)
function _dual_boundary(disc::FerriteDiscretization, fm::ForwardModel, Z)
    model, nu = fm.model, ndofs_u(disc)
    if model isa GapModel
        lens = [electrode_length(disc, e) for e in model.measure]
        Πt = _remove_mean!(Matrix(1.0I, n_measure(fm), n_measure(fm)))
        return _ElectrodeCurrents(_facet_map(model.measure), lens, Πt)
    elseif model isa CompleteElectrodeModel
        return _Robin(_facet_map(model.electrodes), model.z, Matrix(Z[nu+1:end, :]))
    end
    return _ZeroFlux(Set{FacetIndex}())           # nodal (point) measurement loads
end

"""
    residual_indicator(disc, fm, σ, X, currents = nothing; mode = :neumann, normalize = false)

Residual-based error indicator of the forward problem, one value `η_K²` per cell, summed over the
patterns (columns of `X`, the states from [`forward_neumann`](@ref) or
[`forward_dirichlet`](@ref)):

    η_K² = Σₛ h_K² ‖∇⋅(σ∇uₛ)‖²_K + ½ Σ_F h_F ‖[σ∂ₙuₛ]‖²_F + Σ_{F ⊂ ∂Ω} h_F ‖gₛ - σ∂ₙuₛ‖²_F.

The boundary current density `gₛ` comes from the electrode model: the current density of the
continuum model and `Iₗ/|eₗ|` of the gap model (both need `currents`), `(Uₗ - u)/zₗ` of the
complete electrode model, zero in the gaps. Point loads are not estimated, and Dirichlet
boundaries (`mode = :dirichlet`) have no boundary term. `normalize = true` scales every
pattern's indicator to unit sum. `√Σ_K η_K²` bounds the energy error up to a constant; it works
on conforming and hanging-node meshes.
"""
function residual_indicator(disc::FerriteDiscretization, fm::ForwardModel, σ::AbstractVector,
                            X::AbstractVecOrMat, currents = nothing; mode::Symbol = :neumann,
                            normalize::Bool = false)
    mode in (:neumann, :dirichlet) || throw(ArgumentError("mode must be :neumann or :dirichlet"))
    Xm = _as_matrix(X)
    bd = _primal_boundary(disc, fm, Xm, currents, mode)
    return _residual_indicator(disc, σ, _lift(disc, Xm[1:ndofs_u(disc), :]), bd, normalize)
end

function _residual_indicator(disc::FerriteDiscretization, σ, U::AbstractMatrix, bd::_BoundaryData, normalize::Bool)
    grid = disc.grid
    dim = Ferrite.getspatialdim(grid)
    CT = getcelltype(grid)
    shape = Ferrite.getrefshape(CT)
    gip = Ferrite.geometric_interpolation(CT)
    ip_u, ip_σ = disc.ip_u, disc.ip_σ
    pu = Ferrite.getorder(ip_u)
    cv_u, cv_σ = disc.cv_u, disc.cv_σ
    cv_h = pu >= 2 ? CellValues(cv_u.qr, ip_u; update_hessians = true) : nothing
    fqr = FacetQuadratureRule{shape}(2pu + 1)
    fv_u, fv_σ = FacetValues(fqr, ip_u), FacetValues(fqr, ip_σ)
    nbu, nbσ = getnbasefunctions(ip_u), getnbasefunctions(ip_σ)
    ncell, s = getncells(grid), size(U, 2)
    ηs = zeros(ncell, s)
    udofs, udofs2 = zeros(Int, nbu), zeros(Int, nbu)
    σdofs, σdofs2 = zeros(Int, nbσ), zeros(Int, nbσ)
    ue, ge = zeros(nbu), zeros(nbu)
    σe, σe2 = zeros(nbσ), zeros(nbσ)
    ∇φ2 = zeros(Vec{dim, Float64}, nbu)
    finder = Ferrite.NewtonLineSearchPointFinder()
    Ggather = bd isa _Density ? bd.G : nothing

    # element residuals ∇⋅(σ∇u) = ∇σ⋅∇u + σΔu
    for cell in CellIterator(disc.dh_u)
        c = cellid(cell)
        reinit!(cv_u, cell)
        reinit!(cv_σ, cell)
        cv_h === nothing || reinit!(cv_h, cell)
        celldofs!(udofs, disc.dh_u, c)
        celldofs!(σdofs, disc.dh_σ, c)
        σe .= view(σ, σdofs)
        vol = sum(q -> getdetJdV(cv_u, q), 1:getnquadpoints(cv_u))
        h2 = vol^(2 / dim)
        for k in 1:s
            ue .= view(U, udofs, k)
            acc = 0.0
            for q in 1:getnquadpoints(cv_u)
                r = function_gradient(cv_σ, q, σe) ⋅ function_gradient(cv_u, q, ue)
                if cv_h !== nothing
                    r += function_value(cv_σ, q, σe) * tr(function_hessian(cv_h, q, ue))
                end
                acc += r^2 * getdetJdV(cv_u, q)
            end
            ηs[c, k] += h2 * acc
        end
    end

    # interior facets: jumps of the normal current density
    for (c1, f1, c2) in disc.interior_facets
        x1, x2 = getcoordinates(grid, c1), getcoordinates(grid, c2)
        reinit!(fv_u, x1, f1)
        reinit!(fv_σ, x1, f1)
        celldofs!(udofs, disc.dh_u, c1)
        celldofs!(udofs2, disc.dh_u, c2)
        celldofs!(σdofs, disc.dh_σ, c1)
        celldofs!(σdofs2, disc.dh_σ, c2)
        σe .= view(σ, σdofs)
        σe2 .= view(σ, σdofs2)
        hF = sum(q -> getdetJdV(fv_u, q), 1:getnquadpoints(fv_u))
        for q in 1:getnquadpoints(fv_u)
            x = spatial_coordinate(fv_u, q, x1)
            ok, ξ = Ferrite.find_local_coordinate(gip, x2, x, finder)
            ok || error("facet point not found in the neighbouring cell $c2")
            J, _ = Ferrite.calculate_jacobian_and_spatial_coordinate(gip, ξ, x2)
            Jinv = inv(J)
            for i in 1:nbu
                ∇φ2[i] = Ferrite.reference_shape_gradient(ip_u, ξ, i) ⋅ Jinv
            end
            σ2 = sum(a -> σe2[a] * Ferrite.reference_shape_value(ip_σ, ξ, a), 1:nbσ)
            σ1 = function_value(fv_σ, q, σe)
            n = getnormal(fv_u, q)
            w = hF * getdetJdV(fv_u, q) / 2
            for k in 1:s
                ue .= view(U, udofs, k)
                ∇u2 = sum(i -> U[udofs2[i], k] * ∇φ2[i], 1:nbu)
                jump = σ1 * (function_gradient(fv_u, q, ue) ⋅ n) - σ2 * (∇u2 ⋅ n)
                ηs[c1, k] += w * jump^2
                ηs[c2, k] += w * jump^2
            end
        end
    end

    # boundary facets: prescribed current density minus the discrete one
    if !(bd isa _NoBoundary)
        for fi in disc.boundary_facets
            c, f = fi.idx
            xc = getcoordinates(grid, c)
            reinit!(fv_u, xc, f)
            reinit!(fv_σ, xc, f)
            celldofs!(udofs, disc.dh_u, c)
            celldofs!(σdofs, disc.dh_σ, c)
            σe .= view(σ, σdofs)
            hF = sum(q -> getdetJdV(fv_u, q), 1:getnquadpoints(fv_u))
            for k in 1:s
                ue .= view(U, udofs, k)
                Ggather === nothing || (ge .= view(Ggather, udofs, k))
                for q in 1:getnquadpoints(fv_u)
                    inc, g = _bdata(bd, fi, k, fv_u, q, ue, ge)
                    inc || continue
                    flux = function_value(fv_σ, q, σe) * (function_gradient(fv_u, q, ue) ⋅ getnormal(fv_u, q))
                    ηs[c, k] += hF * (g - flux)^2 * getdetJdV(fv_u, q)
                end
            end
        end
    end

    if normalize
        for k in 1:s
            t = sum(view(ηs, :, k))
            t > 0 && (view(ηs, :, k) ./= t)
        end
    end
    return vec(sum(ηs; dims = 2))
end
