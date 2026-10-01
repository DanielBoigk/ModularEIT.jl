# Coefficient vectors of functions x ↦ f(x) in the u or σ space: nodal interpolation and
# L² projection.

_field_dh(d::FerriteDiscretization, field::Symbol) =
    field === :u ? d.dh_u : field === :σ ? d.dh_σ : throw(ArgumentError("field must be :u or :σ, got :$field"))
_field_cv(d::FerriteDiscretization, field::Symbol) =
    field === :u ? d.cv_u : field === :σ ? d.cv_σ : throw(ArgumentError("field must be :u or :σ, got :$field"))

"""
    interpolate_function(disc, f; field = :σ)

Coefficients of the interpolant of `f(x)` (`x` a Ferrite `Vec`) in the `:u` or `:σ` space:
`f` evaluated at the nodes of the interpolation (for piecewise constants: at the cell
centroids).
"""
function interpolate_function(d::FerriteDiscretization, f; field::Symbol = :σ)
    dh = _field_dh(d, field)
    a = zeros(ndofs(dh))
    apply_analytical!(a, dh, field, x -> f(x))      # Ferrite needs a Function (phantoms are callable structs)
    return field === :u ? _restrict(d, a) : a
end

# Ferrite's default triangle rules (Dunavant) end at order 8; Gauss–Jacobi covers 9–15
_quadrature_rule(shape, order) =
    shape === RefTriangle && order > 8 ? QuadratureRule{shape}(:gaussjacobi, order) : QuadratureRule{shape}(order)

"""
    l2_project(disc, f; field = :σ, mats = nothing, quadrature_order = nothing)

Coefficients of the L² projection of `f(x)` onto the `:u` or `:σ` space, `M a = (∫ f φᵢ)ᵢ`.
Pass `mats = FEMatrices(disc)` to reuse assembled mass matrices. `quadrature_order` sets the
rule for `∫ f φᵢ` (default: the rule of the discretization), e.g. higher for rough `f`.
"""
function l2_project(d::FerriteDiscretization, f; field::Symbol = :σ, mats = nothing, quadrature_order = nothing)
    dh, cv = _field_dh(d, field), _field_cv(d, field)
    if quadrature_order !== nothing
        shape = Ferrite.getrefshape(getcelltype(d.grid))
        cv = CellValues(_quadrature_rule(shape, quadrature_order), field === :u ? d.ip_u : d.ip_σ)
    end
    b = zeros(ndofs(dh))
    n = getnbasefunctions(cv)
    be = zeros(n)
    for cell in CellIterator(dh)
        reinit!(cv, cell)
        fill!(be, 0)
        x = getcoordinates(cell)
        for q in 1:getnquadpoints(cv)
            fq = f(spatial_coordinate(cv, q, x)) * getdetJdV(cv, q)
            for i in 1:n
                be[i] += fq * shape_value(cv, q, i)
            end
        end
        assemble!(b, celldofs(cell), be)
    end
    if field === :u
        b = _condense(d, b)
    end
    if mats === nothing
        M = assemble_mass(dh, _field_cv(d, field))
        M = field === :u ? _condense(d, M) : M
        return cholesky(Symmetric(M)) \ b
    end
    return field === :σ ? mats.M_σ_fac \ b : cholesky(Symmetric(mats.M_u)) \ b
end
