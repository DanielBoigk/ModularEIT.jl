using Ferrite
export assemble_function_vector

function assemble_function_vector(fe::FerriteFESpace, f, M_cholesky)
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
    return M_cholesky \ F
end
