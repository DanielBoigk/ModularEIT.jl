using Ferrite: Ferrite, Grid, getcells, getnnodes, get_node_coordinate, getcelltype
using Gridap: Gridap, VectorValue
using Gridap.Geometry: UnstructuredGrid, UnstructuredDiscreteModel, NonOriented
using Gridap.ReferenceFEs: LagrangianRefFE, TRI, QUAD
using Gridap.Arrays: Table

# the Ferrite grid as a Gridap model with the same nodes and cells (same order)
function gridap_model(grid::Grid)
    nodes = [VectorValue(Tuple(get_node_coordinate(grid, i))) for i in 1:getnnodes(grid)]
    C = getcelltype(grid)
    if C === Ferrite.Triangle
        cells = Table([collect(c.nodes) for c in getcells(grid)])
        reffe = LagrangianRefFE(Float64, TRI, 1)
    elseif C === Ferrite.Quadrilateral
        # Ferrite: counter-clockwise; Gridap: lexicographic
        cells = Table([collect(c.nodes)[[1, 2, 4, 3]] for c in getcells(grid)])
        reffe = LagrangianRefFE(Float64, QUAD, 1)
    else
        error("unsupported cell type $C")
    end
    g = UnstructuredGrid(nodes, cells, [reffe], fill(Int8(1), length(cells)), NonOriented())
    return UnstructuredDiscreteModel(g)
end

# permutation p with dofs of `a` at the coordinates of dofs of `b`: a_dof[p[k]] ↔ b_dof[k]
function dof_permutation(xa, ya, xb, yb)
    key(x, y) = (round(x; digits = 9), round(y; digits = 9))
    pos = Dict(key(x, y) => i for (i, (x, y)) in enumerate(zip(xa, ya)))
    return [pos[key(x, y)] for (x, y) in zip(xb, yb)]
end
