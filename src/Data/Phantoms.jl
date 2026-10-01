# Synthetic conductivities as functions x ↦ σ(x), independent of any mesh (wiki: Synthetic
# Conductivity Data, Inverse Crime, Spectral Image Corruption, Spectral Sobolev Norms on
# Rectangles). They are put on a discretization by `conductivity(disc, phantom)` (L² projection,
# i.e. cell averages for piecewise constants), so data can be simulated on a finer mesh than the
# one used for the reconstruction.
#
# Gaussian random fields and the image corruption work in the cosine basis of a pixel grid (the
# eigenvectors of the Neumann Laplacian, DCT-II). Transforms are dense matrix products, which is
# cheap at image sizes (n³ for n × n pixels).

# ---------------------------------------------------------------------------------------------
# Inclusions

_point(p) = (Float64(p[1]), Float64(p[2]))

"""
    CircleInclusion(center, radius, value)

Disc inclusion with conductivity `value`.
"""
struct CircleInclusion <: AbstractInclusion
    center::NTuple{2, Float64}
    radius::Float64
    value::Float64
end
CircleInclusion(center, radius::Real, value::Real) = CircleInclusion(_point(center), Float64(radius), Float64(value))
Base.in(x, c::CircleInclusion) = (x[1] - c.center[1])^2 + (x[2] - c.center[2])^2 <= c.radius^2

"""
    EllipseInclusion(center, (a, b), angle, value)

Elliptic inclusion with semi-axes `a` (along the direction `angle`) and `b`.
"""
struct EllipseInclusion <: AbstractInclusion
    center::NTuple{2, Float64}
    semiaxes::NTuple{2, Float64}
    angle::Float64
    value::Float64
end
EllipseInclusion(center, semiaxes, angle::Real, value::Real) =
    EllipseInclusion(_point(center), _point(semiaxes), Float64(angle), Float64(value))
function Base.in(x, e::EllipseInclusion)
    dx, dy = x[1] - e.center[1], x[2] - e.center[2]
    c, s = cos(e.angle), sin(e.angle)
    u, v = c * dx + s * dy, -s * dx + c * dy
    return (u / e.semiaxes[1])^2 + (v / e.semiaxes[2])^2 <= 1
end

"""
    PolygonInclusion(vertices, value)

Polygonal inclusion (simple polygon, vertices in order) with conductivity `value`.
"""
struct PolygonInclusion <: AbstractInclusion
    vertices::Vector{NTuple{2, Float64}}
    value::Float64
end
PolygonInclusion(vertices, value::Real) = PolygonInclusion([_point(v) for v in vertices], Float64(value))
Base.:(==)(a::PolygonInclusion, b::PolygonInclusion) = a.vertices == b.vertices && a.value == b.value
function Base.in(x, p::PolygonInclusion)                    # even–odd rule
    inside = false
    V = p.vertices
    j = length(V)
    for i in eachindex(V)
        (xi, yi), (xj, yj) = V[i], V[j]
        if (yi > x[2]) != (yj > x[2]) && x[1] < (xj - xi) * (x[2] - yi) / (yj - yi) + xi
            inside = !inside
        end
        j = i
    end
    return inside
end

# centre and radius of a bounding circle
_center(c::CircleInclusion) = c.center
_center(e::EllipseInclusion) = e.center
_center(p::PolygonInclusion) = (sum(first, p.vertices) / length(p.vertices), sum(last, p.vertices) / length(p.vertices))
_bounding_radius(c::CircleInclusion) = c.radius
_bounding_radius(e::EllipseInclusion) = maximum(e.semiaxes)
_bounding_radius(p::PolygonInclusion) = (c = _center(p); maximum(v -> hypot(v[1] - c[1], v[2] - c[2]), p.vertices))

_translate(c::CircleInclusion, d) = CircleInclusion(c.center .+ d, c.radius, c.value)
_translate(e::EllipseInclusion, d) = EllipseInclusion(e.center .+ d, e.semiaxes, e.angle, e.value)
_translate(p::PolygonInclusion, d) = PolygonInclusion([v .+ d for v in p.vertices], p.value)

"""
    InclusionPhantom(background, inclusions)

Piecewise constant conductivity: `background` outside all inclusions, the value of the last
inclusion containing `x` otherwise (later inclusions are painted on top). Callable: `ph(x)`.
"""
struct InclusionPhantom{I <: AbstractInclusion}
    background::Float64
    inclusions::Vector{I}
end
InclusionPhantom(background::Real, inclusions::AbstractVector) =
    InclusionPhantom(Float64(background), collect(AbstractInclusion, inclusions))
function (ph::InclusionPhantom)(x)
    v = ph.background
    for inc in ph.inclusions
        x in inc && (v = inc.value)
    end
    return v
end

"""
    random_inclusions(rng = Random.default_rng(); count = 1:3, domain = :disk, center = (0, 0),
                      radius = 1, bbox = (-1, 1, -1, 1), sizes = (0.1, 0.35), values = (0.2, 5.0),
                      background = 1, shapes = (:circle, :ellipse, :polygon), margin = 0.05,
                      separation = 0.05, maxtries = 1000)

Random [`InclusionPhantom`](@ref): `rand(rng, count)` non-overlapping inclusions of random
shape (circles, ellipses with random orientation and aspect ratio, star-shaped polygons with 3–7
vertices) inside the domain (`:disk` with `center` and `radius`, or `:box` with
`bbox = (xmin, xmax, ymin, ymax)`). Sizes (bounding radii) are drawn from `sizes` relative to
the domain half-width, values log-uniformly from `values`. Every inclusion keeps the distance
`margin` from the boundary and `separation` from the others; an inclusion that cannot be placed
in `maxtries` attempts is dropped.
"""
function random_inclusions(rng::AbstractRNG = Random.default_rng(); count = 1:3, domain::Symbol = :disk,
                           center = (0.0, 0.0), radius::Real = 1.0, bbox = (-1.0, 1.0, -1.0, 1.0),
                           sizes = (0.1, 0.35), values = (0.2, 5.0), background::Real = 1.0,
                           shapes = (:circle, :ellipse, :polygon), margin::Real = 0.05,
                           separation::Real = 0.05, maxtries::Integer = 1000)
    domain in (:disk, :box) || throw(ArgumentError("domain must be :disk or :box, got :$domain"))
    all(in((:circle, :ellipse, :polygon)), shapes) || throw(ArgumentError("unknown shape in $shapes"))
    0 < values[1] <= values[2] || throw(ArgumentError("values must be a positive range"))
    half = domain === :disk ? Float64(radius) : min(bbox[2] - bbox[1], bbox[4] - bbox[3]) / 2
    placed = AbstractInclusion[]
    for _ in 1:rand(rng, count)
        value = exp(log(values[1]) + rand(rng) * (log(values[2]) - log(values[1])))
        for _ in 1:maxtries
            r = half * (sizes[1] + rand(rng) * (sizes[2] - sizes[1]))
            shape = _random_shape(rng, rand(rng, collect(shapes)), r, value)
            shape = _translate(shape, .-_center(shape))               # bounding centre at the origin
            rb = _bounding_radius(shape)
            c = _random_center(rng, domain, center, radius, bbox, rb + margin)
            c === nothing && continue
            all(placed) do other
                hypot((c .- _center(other))...) >= rb + _bounding_radius(other) + separation
            end || continue
            push!(placed, _translate(shape, c))
            break
        end
    end
    return InclusionPhantom(Float64(background), placed)
end

function _random_shape(rng, kind, r, value)
    kind === :circle && return CircleInclusion((0.0, 0.0), r, value)
    kind === :ellipse && return EllipseInclusion((0.0, 0.0), (r, r * (0.3 + 0.7rand(rng))), π * rand(rng), value)
    k = rand(rng, 3:7)
    θ = sort(2π .* rand(rng, k))
    return PolygonInclusion([(ρ * cos(t), ρ * sin(t)) for (t, ρ) in zip(θ, r .* (0.5 .+ 0.5 .* rand(rng, k)))], value)
end

# uniform centre such that the disc of radius `rb` lies in the domain (nothing if impossible)
function _random_center(rng, domain, center, radius, bbox, rb)
    if domain === :disk
        ρ = radius - rb
        ρ > 0 || return nothing
        t, s = 2π * rand(rng), ρ * sqrt(rand(rng))
        return (center[1] + s * cos(t), center[2] + s * sin(t))
    end
    xmin, xmax, ymin, ymax = bbox
    (xmax - xmin > 2rb && ymax - ymin > 2rb) || return nothing
    return (xmin + rb + rand(rng) * (xmax - xmin - 2rb), ymin + rb + rand(rng) * (ymax - ymin - 2rb))
end

# ---------------------------------------------------------------------------------------------
# Pixel functions, images, random fields

# Sampling of a pixel image at a point of the bounding box (xmin, xmax, ymin, ymax).
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

"""
    PixelFunction(values; bbox = (-1, 1, -1, 1), interpolation = :nearest)

Function on the rectangle `bbox = (xmin, xmax, ymin, ymax)` given by pixel values (row 1 at the
top, as for `to_image` (Ferrite back end)), evaluated by `:nearest` pixel or `:bilinear` interpolation
between pixel centres. Points outside the box take the value of the nearest border pixel.
"""
struct PixelFunction
    values::Matrix{Float64}
    bbox::NTuple{4, Float64}
    interpolation::Symbol
    function PixelFunction(values::AbstractMatrix; bbox = (-1.0, 1.0, -1.0, 1.0), interpolation::Symbol = :nearest)
        interpolation in (:nearest, :bilinear) ||
            throw(ArgumentError("interpolation must be :nearest or :bilinear, got :$interpolation"))
        all(isfinite, values) || throw(ArgumentError("pixel values must be finite"))
        return new(Matrix{Float64}(values), Tuple(Float64.(bbox)), interpolation)
    end
end
function (f::PixelFunction)(x)
    xmin, xmax, ymin, ymax = f.bbox
    p = (clamp(x[1], xmin, xmax), clamp(x[2], ymin, ymax))
    return f.interpolation === :nearest ? _pixel_value(f.values, f.bbox, p) : _bilinear(f.values, f.bbox, p)
end

"""
    image_phantom(img, σmin, σmax; bbox = (-1, 1, -1, 1), interpolation = :nearest)

Conductivity from a grayscale image with intensities in `[0, 1]`, mapped affinely to
`[σmin, σmax]` (`0 < σmin < σmax`, so the forward problem stays elliptic), as a
[`PixelFunction`](@ref) on `bbox`.
"""
function image_phantom(img::AbstractMatrix, σmin::Real, σmax::Real; bbox = (-1.0, 1.0, -1.0, 1.0),
                       interpolation::Symbol = :nearest)
    0 < σmin < σmax || throw(ArgumentError("need 0 < σmin < σmax"))
    all(v -> 0 <= v <= 1, img) || throw(ArgumentError("image intensities must lie in [0, 1]"))
    return PixelFunction(σmin .+ (σmax - σmin) .* img; bbox, interpolation)
end

"""
    TransformedPhantom(f, g)

The function `x ↦ g(f(x))`, e.g. a conductivity derived from a random field; see
[`lognormal_phantom`](@ref) and [`levelset_phantom`](@ref).
"""
struct TransformedPhantom{F, G}
    f::F
    g::G
end
(t::TransformedPhantom)(x) = t.g(t.f(x))

"""
    lognormal_phantom(field; σ0 = 1, s = 0.5)

Log-normal conductivity `σ0 exp(s f(x))` from a (Gaussian random) field `f`.
"""
lognormal_phantom(field; σ0::Real = 1.0, s::Real = 0.5) = TransformedPhantom(field, v -> σ0 * exp(s * v))

"""
    levelset_phantom(field; level = 0, inside = 2, outside = 1)

Two-phase conductivity: `inside` where `f(x) > level`, `outside` elsewhere. For a Gaussian
random field this gives random inclusions with smooth, irregular boundaries.
"""
levelset_phantom(field; level::Real = 0.0, inside::Real = 2.0, outside::Real = 1.0) =
    TransformedPhantom(field, v -> v > level ? Float64(inside) : Float64(outside))

# orthonormal DCT-II matrix: C[k+1, j+1] = sₖ cos(π k (j + ½) / n)
function _dct_matrix(n::Integer)
    C = [cos(π * k * (j + 0.5) / n) for k in 0:(n - 1), j in 0:(n - 1)]
    C[1, :] .*= sqrt(1 / n)
    C[2:end, :] .*= sqrt(2 / n)
    return C
end

"""
    gaussian_random_field(rng = Random.default_rng(); bbox = (-1, 1, -1, 1), ℓ = 0.2, order = 2,
                          pixel = ℓ / 10, pad = 3ℓ)

Sample of a Gaussian random field with covariance `(1 - ℓ²Δ)^(-order)` (Matérn type: length
scale `ℓ`, smoothness `ν = order - 1` in 2D; `order = 2` is the Whittle field, `order = 1.5` the
exponential covariance), scaled to unit marginal variance on average over `bbox`. Sampled on a
pixel grid (size `pixel`) over `bbox` enlarged by `pad` in the cosine basis of the Neumann
Laplacian, which keeps the reflecting boundary away from the box. Returns a bilinear
[`PixelFunction`](@ref); combine with [`lognormal_phantom`](@ref) or [`levelset_phantom`](@ref).
"""
function gaussian_random_field(rng::AbstractRNG = Random.default_rng(); bbox = (-1.0, 1.0, -1.0, 1.0),
                               ℓ::Real = 0.2, order::Real = 2.0, pixel::Real = ℓ / 10, pad::Real = 3ℓ)
    ℓ > 0 && order > 1 && pixel > 0 || throw(ArgumentError("need ℓ > 0, order > 1 (in 2D) and pixel > 0"))
    xmin, xmax, ymin, ymax = bbox .+ (-pad, pad, -pad, pad)
    nx, ny = ceil(Int, (xmax - xmin) / pixel), ceil(Int, (ymax - ymin) / pixel)
    max(nx, ny) <= 2048 || throw(ArgumentError("grid of $ny × $nx pixels is too large; increase ℓ or pixel"))
    hx, hy = (xmax - xmin) / nx, (ymax - ymin) / ny
    μx = [(2 / hx)^2 * sin(π * k / (2nx))^2 for k in 0:(nx - 1)]      # Neumann Laplacian eigenvalues
    μy = [(2 / hy)^2 * sin(π * k / (2ny))^2 for k in 0:(ny - 1)]
    S = [(1 + ℓ^2 * (a + b))^(-order) for a in μy, b in μx]           # spectral covariance
    Cx, Cy = _dct_matrix(nx), _dct_matrix(ny)
    F = Cy' * (sqrt.(S) .* randn(rng, ny, nx)) * Cx                   # values, row 1 = top
    # marginal variances Σ S_kl C[k,i]² C[l,j]², averaged over the pixels inside bbox
    Vmap = (Cy .^ 2)' * S * (Cx .^ 2)
    rows = [i for i in 1:ny if bbox[3] <= ymax - (i - 0.5) * hy <= bbox[4]]
    cols = [j for j in 1:nx if bbox[1] <= xmin + (j - 0.5) * hx <= bbox[2]]
    F ./= sqrt(sum(Vmap[rows, cols]) / (length(rows) * length(cols)))
    return PixelFunction(F; bbox = (xmin, xmax, ymin, ymax), interpolation = :bilinear)
end

"""
    corrupt_image(img; rng = Random.default_rng(), steps = 1, spatial_noise = 0.05,
                  spectral_noise = 0.01, damping = 1e-3, exponent = 1)

Synthetic degradation of an image for training denoisers (wiki: *Spectral Image Corruption*).
Repeats `steps` times: add white noise (`spatial_noise`), transform to the orthonormal DCT-II
basis, add frequency-dependent noise `spectral_noise (1 + |ω|²) ξ`, damp by
`exp(-damping |ω|^(2 exponent))`, transform back. `|ω|²` are the eigenvalues of the discrete
Neumann Laplacian of the pixel grid, scaled to `≈ k² + l²` for low frequencies `(k, l)`, so
damping alone (`exponent = 1`) is the discrete heat semigroup: it preserves the mean and
satisfies the maximum principle.
"""
function corrupt_image(img::AbstractMatrix; rng::AbstractRNG = Random.default_rng(), steps::Integer = 1,
                       spatial_noise::Real = 0.05, spectral_noise::Real = 0.01, damping::Real = 1e-3,
                       exponent::Real = 1.0)
    x = Matrix{Float64}(img)
    steps == 0 && return x
    n, m = size(x)
    Cn, Cm = _dct_matrix(n), _dct_matrix(m)
    # |ω|²: eigenvalues of the discrete Neumann Laplacian, scaled to ≈ k² + l² at low frequencies
    ω(k, n) = (2n / π)^2 * sin(π * k / (2n))^2
    ω2 = [ω(k, n) + ω(l, m) for k in 0:(n - 1), l in 0:(m - 1)]
    damp = exp.(-damping .* ω2 .^ exponent)
    for _ in 1:steps
        x .+= spatial_noise .* randn(rng, n, m)
        C = Cn * x * Cm'
        C .+= spectral_noise .* (1 .+ ω2) .* randn(rng, n, m)
        C .*= damp
        x = Cn' * C * Cm
    end
    return x
end
