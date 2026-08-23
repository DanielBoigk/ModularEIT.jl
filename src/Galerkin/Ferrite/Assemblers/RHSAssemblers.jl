using Ferrite
export project_function_to_fem, assemble_rhs_func



"""
    project_function_to_fem(fe, f; space=:u)

L² projection of `f` (a function of physical coordinates) onto an FE space of
`fe`. `space=:u` (default) projects onto the potential field's space
(`fe.dh`/`fe.cellvalues`, reusing the cached `fe.M_fac`); `space=:σ` projects
onto the conductivity field's space (`fe.dh_σ`/`fe.cellvalues_σ`), which may
have a different (e.g. piecewise-constant) order.
"""
function project_function_to_fem(fe::FerriteFESpace, f; space::Symbol=:u)
    if space === :u
        cellvalues, dh, n, M_fac = fe.cellvalues, fe.dh, fe.n, fe.M_fac
    elseif space === :σ
        cellvalues, dh, n = fe.cellvalues_σ, fe.dh_σ, fe.n_σ
        _, M_fac = assemble_M(dh, cellvalues)
    else
        throw(ArgumentError("space must be :u or :σ, got $space"))
    end

    F = zeros(n)
    n_basefuncs = getnbasefunctions(cellvalues)
    Fe = zeros(n_basefuncs)

    for cell in CellIterator(dh)
        fill!(Fe, 0.0)
        reinit!(cellvalues, cell)
        coords = getcoordinates(cell)
        cdofs = celldofs(cell)
        for q in 1:getnquadpoints(cellvalues)
            x_q = spatial_coordinate(cellvalues, q, coords)
            f_val = f(x_q)
            dΩ = getdetJdV(cellvalues, q)

            for i in 1:n_basefuncs
                Fe[i] += f_val * shape_value(cellvalues, q, i) * dΩ
            end
        end
        assemble!(F, cdofs, Fe)
    end
    return M_fac \ F
end

# This assembles ∫(g*v)d∂Ω
function assemble_rhs_func(facetvalues::FacetValues, dh::DofHandler, g_func, ∂Ω)
    f = zeros(ndofs(dh))
    fe = zeros(ndofs_per_cell(dh))
    for facet in FacetIterator(dh, ∂Ω)
        fill!(fe, 0.0)
        reinit!(facetvalues, facet)
        coords = getcoordinates(facet)
        dofs = celldofs(facet)
        for q_point in 1:getnquadpoints(facetvalues)
            x = spatial_coordinate(facetvalues, q_point, coords)
            g = g_func(x)
            dΓ = getdetJdV(facetvalues, q_point)
            for i in 1:getnbasefunctions(facetvalues)
                ϕᵢ = shape_value(facetvalues, q_point, i)
                fe[i] += ϕᵢ * g * dΓ
            end
        end
        assemble!(f, dofs, fe)
    end
    return f
end

function assemble_rhs_func(fe::FerriteFESpace, g_func)
    vec = assemble_rhs_func(fe.facetvalues, fe.dh, g_func, fe.∂Ω)
    up! = fe.up!
    down = fe.down
    b = down(vec)
    mean = Statistics.mean(b)
    b .-= mean
    up!(vec,b)
end


# Write assembler for dirichlet boundary from function
