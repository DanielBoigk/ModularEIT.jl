# Pixel parametrizations: the unknowns of a reconstruction are the values of an n × m pixel grid on a
# rectangle containing the domain, independent of the finite element mesh (which may be refined
# anywhere, e.g. towards the boundary). The conductivity coefficients are weighted averages of the
# pixel values,
#
#     σₐ = Σₖ Pₐₖ θₖ,    Pₐₖ = ∫ φₐ χₖ / ∫ φₐ     (χₖ: indicator of pixel k),
#
# the lumped L² projection of the pixel function onto the σ space: positive, reproducing constants
# (so bounds on pixels bound σ), exact on meshes whose cells lie inside single pixels. The
# integrals are sampled with a quadrature rule that resolves the pixels (as in from_image).
#
# The pixels form their own P0 discretization (`pixel_disc`), so pixel regularizers (TV,
# Tikhonov), images (to_image, unit_image) and transforms work on θ directly.

"""
    PixelParametrization(disc, n, m; bbox = nothing, pixels = :square, quadrature_order = nothing)

Conductivity on `disc` parametrized by the values `θ` of an `n × m` pixel grid on the rectangle
`bbox` (default: the bounding box of the mesh, widened to square pixels unless
`pixels = :stretch`, as in [`unit_image`](@ref)): `σ = P θ` with `P` the lumped L² projection of
the pixel function onto the σ space (weighted pixel averages; positive, constants preserved,
exact on pixel-aligned meshes).

Fields: `P` (`ndofs_σ × n m`), `pixel_disc` (P0 discretization of the pixel grid; `θ` are its
coefficients, e.g. for [`TotalVariationRegularizer`](@ref)), `active` (pixels that influence σ),
`centres` (pixel centres in parameter order), `bbox`, `n`, `m`.

See [`conductivity`](@ref), [`pixel_image`](@ref), [`pixel_parameters`](@ref),
[`ParametrizedObjective`](@ref), [`SubspaceParametrization`](@ref).
"""
struct PixelParametrization{D, PD} <: AbstractParametrization
    disc::D
    pixel_disc::PD
    n::Int
    m::Int
    bbox::NTuple{4, Float64}
    P::SparseMatrixCSC{Float64, Int}
    active::BitVector
    centres::Vector{Vec{2, Float64}}
    image_index::Vector{Int}          # parameter k → linear index into the n × m image
end

function PixelParametrization(d::FerriteDiscretization, n::Integer, m::Integer; bbox = nothing,
                              pixels::Symbol = :square, quadrature_order = nothing)
    Ferrite.getspatialdim(d.grid) == 2 || throw(ArgumentError("pixel parametrizations need a 2D mesh"))
    n > 0 && m > 0 || throw(ArgumentError("the pixel grid must not be empty"))
    pixels in (:square, :stretch) || throw(ArgumentError("pixels must be :square or :stretch, got :$pixels"))
    box = bbox === nothing ? _bounding_box(d.grid) : NTuple{4, Float64}(bbox)
    pixels === :square && (box = _square_pixel_box(box, n, m))
    xmin, xmax, ymin, ymax = box
    Δx, Δy = (xmax - xmin) / m, (ymax - ymin) / n
    pgrid = generate_grid(Quadrilateral, (m, n), Vec(xmin, ymin), Vec(xmax, ymax))
    pdisc = FerriteDiscretization(pgrid)
    # parameter numbering = σ dofs of the pixel discretization; locate each pixel (row i from the top)
    npix = n * m
    param_of = zeros(Int, n, m)
    centres = Vector{Vec{2, Float64}}(undef, npix)
    image_index = zeros(Int, npix)
    for c in 1:getncells(pgrid)
        x = getcoordinates(pgrid, c)
        ctr = sum(x) / length(x)
        i = clamp(floor(Int, (ymax - ctr[2]) / Δy) + 1, 1, n)
        j = clamp(floor(Int, (ctr[1] - xmin) / Δx) + 1, 1, m)
        k = only(celldofs(pdisc.dh_σ, c))
        param_of[i, j] = k
        centres[k] = ctr
        image_index[k] = LinearIndices((n, m))[i, j]
    end
    # B[a, k] = ∫ φₐ χₖ by quadrature on the cells of disc
    order = quadrature_order === nothing ? _pixel_quadrature_order(d.grid, min(Δx, Δy)) : quadrature_order
    shape = Ferrite.getrefshape(getcelltype(d.grid))
    cv = CellValues(_quadrature_rule(shape, order), d.ip_σ)
    nb = getnbasefunctions(cv)
    Is, Js, Vs = Int[], Int[], Float64[]
    for cell in CellIterator(d.dh_σ)
        reinit!(cv, cell)
        x = getcoordinates(cell)
        dofs = celldofs(cell)
        for q in 1:getnquadpoints(cv)
            xq = spatial_coordinate(cv, q, x)
            i = clamp(floor(Int, (ymax - xq[2]) / Δy) + 1, 1, n)
            j = clamp(floor(Int, (xq[1] - xmin) / Δx) + 1, 1, m)
            k = param_of[i, j]
            dV = getdetJdV(cv, q)
            for a in 1:nb
                v = shape_value(cv, q, a) * dV
                iszero(v) && continue
                push!(Is, dofs[a]); push!(Js, k); push!(Vs, v)
            end
        end
    end
    B = sparse(Is, Js, Vs, ndofs_σ(d), npix)
    rowsum = vec(sum(B; dims = 2))
    all(>(0), rowsum) || throw(ArgumentError("some σ basis functions are not covered by the pixel grid"))
    P = sparse(Diagonal(1 ./ rowsum) * B)
    active = vec(sum(B; dims = 1)) .> 0
    return PixelParametrization(d, pdisc, Int(n), Int(m), box, P, BitVector(active), centres, image_index)
end

"""
    pixel_image(par, θ)

The pixel values of the parameters `θ` as an `n × m` image (row 1 at the top); pixels that do not
influence the conductivity are `NaN`.
"""
function pixel_image(pp::PixelParametrization, θ::AbstractVector)
    length(θ) == pp.n * pp.m || throw(DimensionMismatch("expected $(pp.n * pp.m) pixel values"))
    img = fill(NaN, pp.n, pp.m)
    for k in eachindex(θ)
        pp.active[k] && (img[pp.image_index[k]] = θ[k])
    end
    return img
end

"""
    pixel_parameters(pp::PixelParametrization, img)

Parameters `θ` of an `n × m` image (row 1 at the top); `NaN` pixels become `fill`
(keyword, default 1).
"""
function pixel_parameters(pp::PixelParametrization, img::AbstractMatrix; fill::Real = 1.0)
    size(img) == (pp.n, pp.m) || throw(DimensionMismatch("image must be $(pp.n) × $(pp.m)"))
    return [isnan(img[pp.image_index[k]]) ? Float64(fill) : Float64(img[pp.image_index[k]]) for k in 1:(pp.n * pp.m)]
end

"""
    dct_basis(pp::PixelParametrization, K)

The `K × K` lowest-frequency orthonormal DCT-II modes of the pixel grid as the columns of an
`n m × K²` matrix (parameter order), sorted by total frequency (the constant mode first). For
[`SubspaceParametrization`](@ref): smooth, low-dimensional conductivities, e.g. as the first stage
of a coarse-to-fine reconstruction.
"""
function dct_basis(pp::PixelParametrization, K::Integer)
    1 <= K <= min(pp.n, pp.m) || throw(ArgumentError("K must lie in 1:$(min(pp.n, pp.m))"))
    Cn, Cm = _dct_matrix(pp.n), _dct_matrix(pp.m)
    modes = sort([(k, l) for k in 0:(K - 1) for l in 0:(K - 1)]; by = kl -> (sum(kl), kl[1]))
    B = zeros(pp.n * pp.m, length(modes))
    for (c, (k, l)) in enumerate(modes)
        img = Cn[k + 1, :] * Cm[l + 1, :]'
        for p in 1:(pp.n * pp.m)
            B[p, c] = img[pp.image_index[p]]
        end
    end
    return B
end

"""
    boundary_band_basis(pp::PixelParametrization, width)

Indicator vectors (sparse columns) of the active pixels whose centre lies within `width` of the
domain boundary: free pixel values along the boundary, where the data determine the conductivity
best. Combined with [`dct_basis`](@ref) (`[C Bb]`) the interior is smooth and low-dimensional and
the boundary band fully resolved.
"""
function boundary_band_basis(pp::PixelParametrization, width::Real)
    d = pp.disc
    segs = map(d.boundary_facets) do fi
        c, f = fi.idx
        a, b = (get_node_coordinate(d.grid, nd) for nd in Ferrite.facets(getcells(d.grid, c))[f])
        (a, b)
    end
    band = [k for k in 1:(pp.n * pp.m)
            if pp.active[k] && minimum(s -> _segment_distance(pp.centres[k], s[1], s[2]), segs) <= width]
    return sparse(band, 1:length(band), ones(length(band)), pp.n * pp.m, length(band))
end
