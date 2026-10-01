# Detection of uniform rectangle grids for the DCT solvers: bilinear u space on a conforming
# quadrilateral mesh whose nodes form a tensor grid with constant spacings and whose cells are the
# grid cells.

"""
    structured_grid(disc; rtol = 1e-8)

[`StructuredGrid`](@ref) of a discretization whose u space is bilinear (Q1) on a uniform
rectangle grid (any spacings `hx`, `hy`), e.g. `generate_grid(Quadrilateral, (nx, ny), ll, ur)`
or a pixel-aligned image mesh. Throws an `ArgumentError` otherwise.
"""
function structured_grid(d::FerriteDiscretization; rtol::Real = 1e-8)
    fail(msg) = throw(ArgumentError("not a uniform rectangle grid of bilinear elements: $msg"))
    getcelltype(d.grid) <: Quadrilateral || fail("cells must be quadrilaterals")
    d.ip_u isa Lagrange{RefQuadrilateral, 1} || fail("the u space must be bilinear (Q1)")
    d.C_u === nothing || fail("hanging nodes are not supported")
    n = ndofs_u(d)
    xs, ys = _dof_coordinates(d, 1:n)
    x0, x1, y0, y1 = extrema(xs)..., extrema(ys)...
    scale = max(x1 - x0, y1 - y0)
    tol = rtol * scale
    ux = _distinct(xs, tol)
    uy = _distinct(ys, tol)
    nx, ny = length(ux), length(uy)
    nx * ny == n || fail("the nodes do not form a tensor grid")
    hx, hy = (x1 - x0) / (nx - 1), (y1 - y0) / (ny - 1)
    all(abs.(diff(ux) .- hx) .<= tol) && all(abs.(diff(uy) .- hy) .<= tol) || fail("the spacing is not uniform")
    perm = zeros(Int, n)
    for dof in 1:n
        i = round(Int, (xs[dof] - x0) / hx)
        j = round(Int, (ys[dof] - y0) / hy)
        abs(xs[dof] - (x0 + i * hx)) <= tol && abs(ys[dof] - (y0 + j * hy)) <= tol || fail("node off the grid")
        p = 1 + i + j * nx
        perm[p] == 0 || fail("two nodes at the same grid point")
        perm[p] = dof
    end
    for c in 1:getncells(d.grid)                       # every cell is one grid cell
        x = getcoordinates(d.grid, c)
        (abs(maximum(p[1] for p in x) - minimum(p[1] for p in x) - hx) <= tol &&
         abs(maximum(p[2] for p in x) - minimum(p[2] for p in x) - hy) <= tol) || fail("cell $c is not a grid cell")
    end
    return StructuredGrid(nx, ny, hx, hy, (x0, y0), perm)
end

# sorted distinct values up to `tol`
function _distinct(v, tol)
    s = sort(v)
    out = [s[1]]
    for x in s
        x - out[end] > tol && push!(out, x)
    end
    return out
end
