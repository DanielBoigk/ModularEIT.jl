# Images

```@meta
CurrentModule = ModularEIT
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
ImageMap
to_image
from_image
```

## Wiki articles

Theory behind this page in the [theory wiki](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/):

- [Pixel Images and Finite Element Functions](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/04-Finite-Elements/Pixel-Images-and-Finite-Element-Functions)
