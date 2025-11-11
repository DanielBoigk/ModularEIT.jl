using Ferrite
export project_function_to_fem, assemble_rhs_func



function project_function_to_fem(fe::FerriteFESpace, f)
    F = zeros(fe.n)
    cellvalues = fe.cellvalues
    dh = fe.dh
    n_basefuncs = getnbasefunctions(cellvalues)
    Fe = zeros(n_basefuncs)
    cdofs = zeros(Int, n_basefuncs)

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
    return fe.M_fac \ F
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
