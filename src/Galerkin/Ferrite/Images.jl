# Finite element functions ↔ pixel images for 2D meshes.
#
# Pixel convention: an n × m image covers the bounding box [xmin, xmax] × [ymin, ymax] (default:
# of the mesh); img[i, j] belongs to the pixel in row i counted from the top and column j counted
# from the left, with centre
#     x_j = xmin + (j - ½) Δx,   y_i = ymax - (i - ½) Δy,   Δx = (xmax - xmin)/m, Δy = (ymax - ymin)/n.
#
# FE → image: evaluation at the pixel centres, one sparse matrix S (pixels × coefficients)
#             precomputed per mesh and image size (ImageMap).
# image → FE: :interpolate  bilinear interpolation of the pixel values at the nodes of the FE
#                           space (cell centroids for piecewise constants); exact for linear
#                           functions, the inverse of S for pixel-aligned meshes;
#             :l2           L² projection of the image, pixels taken as constant on their area.
# On quadrilateral meshes whose cells are the pixels (generate_grid(Quadrilateral, (m, n))),
# both directions are exact inverses for piecewise constant σ.

"""
    ImageMap(disc, n, m; field = :σ, bbox = nothing)

Precomputed map between the `field` (`:u` or `:σ`) of a 2D discretization and `n × m` images
(`n` rows along y, `m` columns along x) on the bounding box `bbox = (xmin, xmax, ymin, ymax)`
(default: bounding box of the mesh). Row 1 is the top of the image. Fields: `S` (sparse,
`n m × ndofs`, pixel centres in column-major order), `inside` (pixels whose centre lies in the
mesh), `nodes` (coordinates of the nodes of the FE space, for sampling images).

See [`to_image`](@ref) and [`from_image`](@ref).
"""
struct ImageMap{D <: FerriteDiscretization}
    disc::D
    field::Symbol
    n::Int
    m::Int
    bbox::NTuple{4, Float64}
    S::SparseMatrixCSC{Float64, Int}
    inside::BitMatrix
    nodes::Vector{Vec{2, Float64}}
end

function ImageMap(disc::FerriteDiscretization, n::Integer, m::Integer; field::Symbol = :σ, bbox = nothing)
    Ferrite.getspatialdim(disc.grid) == 2 || throw(ArgumentError("images need a 2D mesh"))
    n > 0 && m > 0 || throw(ArgumentError("image size must be positive"))
    dh = _field_dh(disc, field)
    ip = field === :u ? disc.ip_u : disc.ip_σ
    box = bbox === nothing ? _bounding_box(disc.grid) : NTuple{4, Float64}(bbox)
    xmin, xmax, ymin, ymax = box
    Δx, Δy = (xmax - xmin) / m, (ymax - ymin) / n
    centres = vec([Vec(xmin + (j - 0.5) * Δx, ymax - (i - 0.5) * Δy) for i in 1:n, j in 1:m])
    ph = PointEvalHandler(disc.grid, centres; warn = false, search_nneighbors = 8)
    Is, Js, Vs = Int[], Int[], Float64[]
    nb = getnbasefunctions(ip)
    dofs = zeros(Int, ndofs_per_cell(dh))
    for (k, (c, ξ)) in enumerate(zip(ph.cells, ph.local_coords))
        c === nothing && continue
        celldofs!(dofs, dh, c)
        for a in 1:nb
            v = Ferrite.reference_shape_value(ip, ξ, a)
            iszero(v) && continue
            push!(Is, k); push!(Js, dofs[a]); push!(Vs, v)
        end
    end
    S = sparse(Is, Js, Vs, n * m, ndofs(dh))
    field === :u && disc.C_u !== nothing && (S = S * disc.C_u)
    inside = reshape(BitVector([c !== nothing for c in ph.cells]), n, m)
    xs = interpolate_function(disc, p -> p[1]; field)
    ys = interpolate_function(disc, p -> p[2]; field)
    return ImageMap(disc, field, Int(n), Int(m), box, S, inside, [Vec(x, y) for (x, y) in zip(xs, ys)])
end

function _bounding_box(grid)
    xs = [get_node_coordinate(grid, i) for i in 1:getnnodes(grid)]
    return (minimum(x -> x[1], xs), maximum(x -> x[1], xs), minimum(x -> x[2], xs), maximum(x -> x[2], xs))
end

"""
    to_image(map::ImageMap, coeffs; outside = NaN)
    to_image(disc, coeffs, n, m; field = :σ, bbox = nothing, outside = NaN)

`n × m` image of the finite element function with coefficients `coeffs`, evaluated at the pixel
centres (row 1 at the top, see [`ImageMap`](@ref)). Pixels outside the mesh get `outside`.
"""
function to_image(im::ImageMap, coeffs::AbstractVector; outside = NaN)
    length(coeffs) == size(im.S, 2) || throw(DimensionMismatch("expected $(size(im.S, 2)) coefficients"))
    img = reshape(im.S * coeffs, im.n, im.m)
    img[.!im.inside] .= outside
    return img
end
to_image(disc::FerriteDiscretization, coeffs::AbstractVector, n::Integer, m::Integer;
         field::Symbol = :σ, bbox = nothing, outside = NaN) =
    to_image(ImageMap(disc, n, m; field, bbox), coeffs; outside)

"""
    from_image(map::ImageMap, img; method = :interpolate, quadrature_order = nothing)
    from_image(disc, img; field = :σ, bbox = nothing, method = :interpolate, quadrature_order = nothing)

Coefficients of a finite element function from an `n × m` image (same pixel convention as
[`to_image`](@ref)):

- `method = :interpolate`: bilinear interpolation of the pixel values at the nodes of the FE
  space (cell centroids for piecewise constants), linearly extrapolated within the outer half
  pixel. Exact for linear functions, and the inverse of `to_image` on pixel-aligned meshes.
- `method = :l2`: L² projection of the image, each pixel constant on its area. The pixels are
  sampled at the points of a quadrature rule; the default order grows with the number of pixels
  per cell (several points per pixel, capped at order 15 on triangles). The integral of the image
  is preserved up to this sampling error (exactly on pixel-aligned meshes). Smooths images that
  are finer than the mesh.

`NaN` pixels (outside the domain) are filled from their nearest finite neighbours first.
"""
function from_image(im::ImageMap, img::AbstractMatrix; method::Symbol = :interpolate, quadrature_order = nothing)
    size(img) == (im.n, im.m) || throw(DimensionMismatch("image must be $(im.n) × $(im.m)"))
    method in (:interpolate, :l2) || throw(ArgumentError("method must be :interpolate or :l2, got :$method"))
    filled = _fill_nan(Matrix{Float64}(img))
    if method === :interpolate
        return [_bilinear(filled, im.bbox, p) for p in im.nodes]
    end
    order = quadrature_order === nothing ? _pixel_quadrature_order(im) : quadrature_order
    return l2_project(im.disc, p -> _pixel_value(filled, im.bbox, p); field = im.field, quadrature_order = order)
end

# quadrature order resolving every pixel by a few points: ~2 points per pixel and direction
function _pixel_quadrature_order(im::ImageMap)
    grid = im.disc.grid
    xmin, xmax, ymin, ymax = im.bbox
    pixel = min((xmax - xmin) / im.m, (ymax - ymin) / im.n)
    h = maximum(1:getncells(grid)) do c
        x = getcoordinates(grid, c)
        maximum(norm(a - b) for a in x for b in x)
    end
    npts = 2 * ceil(Int, h / pixel)                       # points per direction and cell
    shape = Ferrite.getrefshape(getcelltype(grid))
    shape <: Ferrite.RefHypercube && return clamp(npts, 2, 64)
    return clamp(npts, 2, 15)                              # simplex: polynomial degree, max. 15
end
from_image(disc::FerriteDiscretization, img::AbstractMatrix; field::Symbol = :σ, bbox = nothing, kwargs...) =
    from_image(ImageMap(disc, size(img, 1), size(img, 2); field, bbox), img; kwargs...)

# continuous pixel coordinates: (row, column) with pixel centres at integers
function _pixel_coords(img, (xmin, xmax, ymin, ymax), p)
    n, m = size(img)
    return (ymax - p[2]) / (ymax - ymin) * n + 0.5, (p[1] - xmin) / (xmax - xmin) * m + 0.5
end

function _bilinear(img, box, p)
    n, m = size(img)
    r, c = _pixel_coords(img, box, p)
    i0 = n == 1 ? 1 : clamp(floor(Int, r), 1, n - 1)
    j0 = m == 1 ? 1 : clamp(floor(Int, c), 1, m - 1)
    s = n == 1 ? 0.0 : r - i0            # may leave [0, 1] at the border: linear extrapolation
    t = m == 1 ? 0.0 : c - j0
    i1, j1 = min(i0 + 1, n), min(j0 + 1, m)
    return (1 - s) * ((1 - t) * img[i0, j0] + t * img[i0, j1]) + s * ((1 - t) * img[i1, j0] + t * img[i1, j1])
end

function _pixel_value(img, box, p)
    n, m = size(img)
    r, c = _pixel_coords(img, box, p)
    return img[clamp(round(Int, r), 1, n), clamp(round(Int, c), 1, m)]
end

# replace NaN pixels by the mean of their finite 8-neighbours, repeatedly (nearest-neighbour fill)
function _fill_nan(img::Matrix{Float64})
    any(isnan, img) || return img
    all(isnan, img) && throw(ArgumentError("the image has no finite pixels"))
    n, m = size(img)
    while any(isnan, img)
        new = copy(img)
        for i in 1:n, j in 1:m
            isnan(img[i, j]) || continue
            acc, cnt = 0.0, 0
            for di in -1:1, dj in -1:1
                ii, jj = i + di, j + dj
                (1 <= ii <= n && 1 <= jj <= m && !isnan(img[ii, jj])) || continue
                acc += img[ii, jj]
                cnt += 1
            end
            cnt > 0 && (new[i, j] = acc / cnt)
        end
        img = new
    end
    return img
end
