using Interpolations
using Images
using ImageIO
using FileIO

export interpolate_from_image
export func_to_array2D
export to_grayscale_image


"""
    interpolate_from_image(img; xinterval=(-1,1), yinterval=(-1,1))

Create a 2D interpolation object from `img`, mapping its pixel indices
onto the coordinate ranges `xinterval` × `yinterval`.

Returns an `Interpolations.ScaledInterpolation`.
"""
function interpolate_from_image(img; xinterval=(-1,1), yinterval=(-1,1))
    img_float = Float64.(img)
    img_float[img_float .== 0] .+= 1e-6

    itp = interpolate(img_float, BSpline(Linear()))
    size_x, size_y = size(img)
    x_range = range(xinterval[1], xinterval[2], length=size_x)
    y_range = range(yinterval[1], yinterval[2], length=size_y)

    return Interpolations.scale(itp, x_range, y_range)
end

using FileIO, ImageIO, ImageMagick  # or ImageIO.jl depending on your setup

"""
    interpolate_from_image(path::String; xinterval=(-1,1), yinterval=(-1,1))

Load an image from `path`, convert to grayscale Float64, and return a
scaled 2D interpolation over the given intervals.
"""
function interpolate_from_image(path::String; xinterval=(-1,1), yinterval=(-1,1))
    img = load(path)
    if ndims(img) == 3
        img = channelview(img)[1, :, :]  # convert RGB → grayscale
    end
    return interpolate_from_image(img; xinterval=xinterval, yinterval=yinterval)
end


"""
    func_to_array2D(f, intervals::Tuple{Tuple{<:Real,<:Real}, Tuple{<:Real,<:Real}}, size::Tuple{Int,Int})

Sample a function `f(x, y)` over an equidistant 2D grid defined by
`intervals = ((x_min, x_max), (y_min, y_max))` and grid size `(nx, ny)`.

Returns an `nx × ny` array of values.
"""
function func_to_array2D(f, intervals::Tuple{Tuple{<:Real,<:Real}, Tuple{<:Real,<:Real}}, size::Tuple{Int,Int})
    (xint, yint) = intervals
    (nx, ny) = size
    x_range = range(xint[1], xint[2]; length=nx)
    y_range = range(yint[1], yint[2]; length=ny)
    return func_to_array2D(f, x_range, y_range)
end

"""
    func_to_array2D(f, x_range, y_range)

Evaluate `f(x, y)` over the Cartesian product of `x_range` and `y_range`.
Returns an array of size `(length(x_range), length(y_range))`.
"""
function func_to_array2D(f, x_range, y_range)
    nx, ny = length(x_range), length(y_range)
    A = Matrix{Float64}(undef, nx, ny)
    @inbounds for j in 1:ny, i in 1:nx
        A[i, j] = f(x_range[i], y_range[j])
    end
    return A
end



function to_grayscale_image(values::AbstractMatrix)
    min_val = minimum(values)
    max_val = maximum(values)
    normalized = (values .- min_val) ./ (max_val - min_val + eps()) # avoid div-by-zero
    img = Gray.(normalized)
    return img
end
