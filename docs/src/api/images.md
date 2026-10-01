# Images

```@meta
CurrentModule = ModularEITFerrite
```

Conversion between finite element functions on 2D meshes and `n × m` pixel images, e.g. for
image-based priors, learned regularisers or plotting. `img[i, j]` is the pixel in row `i` from
the top and column `j` from the left of the bounding box. On quadrilateral meshes whose cells are
the pixels (`generate_grid(Quadrilateral, (m, n))` with piecewise constant σ), both directions
are exact inverses. Theory: wiki article [Pixel Images and Finite Element Functions](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/04-Finite-Elements/Pixel-Images-and-Finite-Element-Functions).

```julia
disc = FerriteDiscretization(grid)
img = to_image(disc, σ, 128, 128)                   # σ coefficients → 128 × 128 image
σ2 = from_image(disc, img)                          # bilinear sampling at the nodes
σ3 = from_image(disc, img; method = :l2)            # L² projection of the pixel image

im = ImageMap(disc, 128, 128; field = :u)           # precompute once, reuse
u_img = to_image(im, u)
```

```@docs
ModularEITFerrite.ImageMap
ModularEITFerrite.to_image
ModularEITFerrite.from_image
```

## Unit images

Fixed-size arrays with values in ``[0, 1]``, for example as network inputs or for data sets: a
rectangle that contains the domain (square pixels by default), a linear or logarithmic value
scaling (automatic, or a fixed range for a whole data set), a fill value and a mask for the
pixels outside the domain, and the way back to coefficients.

```julia
ui = unit_image(disc, σ, 64, 64)                            # auto range, square pixels, 0 outside
ui = unit_image(disc, σ, 64, 64; range = (0.1, 10.0), scale = :log)
ui.image, ui.mask, ui.bbox, ui.range
σ_back = from_unit_image(disc, ui)                          # e.g. after denoising ui.image
```

```@docs
ModularEITFerrite.UnitImage
ModularEITFerrite.unit_image
ModularEITFerrite.from_unit_image
```

## Wiki articles

Theory behind this page in the [theory wiki](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/):

- [Pixel Images and Finite Element Functions](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/04-Finite-Elements/Pixel-Images-and-Finite-Element-Functions)
