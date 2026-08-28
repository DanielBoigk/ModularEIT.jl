using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using Ferrite
using ModularEIT
using Images, FileIO
using LinearAlgebra
using Statistics
using Distributions
using Random
using Plots
using JLD2

using Lux, Reactant, Enzyme, NNlib
using Optimisers, ComponentArrays

# =============================================================================
# initialize.jl
#
# Shared setup for physics-informed diffusion reconstruction of EIT images
# (DPS, and later Diff-PIR / RED-Diff). Two independent pieces get wired
# together here:
#
#   1. The forward EIT model  (same pipeline as 001SolveTest.ipynb: load an
#      image -> FEM grid -> boundary modes G -> DtN map Λ -> add noise -> SVD
#      mode reduction -> f(σ)/∂f(σ), the physics data-fidelity loss and its
#      exact adjoint gradient in FEM conductivity (σ) space).
#
#   2. A pretrained VP-SDE score/noise-predictor network (CNN or U-Net, from
#      ../CNN/ or ../Unet/) that acts as a generic natural-image prior over
#      64x64 grayscale images — reused here as a prior over conductivity maps,
#      following the same "log-conductivity looks like a natural image" idea
#      from 001SolveTest.ipynb's cb00f750 cell.
#
# The two live on different grids (fe.n_σ FEM cells vs. a 64x64 pixel image),
# so `sigma_from_image`/`conductivity_from_sigma` below are the glue between them.
# Reconstruction scripts (dps_reconstruct.jl, ...) `include(joinpath(@__DIR__,
# "initialize.jl"))` and then only need to write the actual sampling loop.
# =============================================================================

# -----------------------------------------------------------------------------
# Parameters — edit these, everything below is derived.
# -----------------------------------------------------------------------------

# --- forward EIT model ---
IMG_PATH   = joinpath(@__DIR__, "..", "reconstructions", "1096.jpg")
GRID_N     = 63     # FEM grid is GRID_N x GRID_N quadrilaterals
NUM_MODES  = 255     # boundary Fourier modes used to build G / Λ
SIGMA_NOISE = 0.0    # relative noise level added to Λ (0 = noiseless); see the
                     # Λn cell in 001SolveTest.ipynb for what this scales

# --- σ-space physics loss f(σ), ∂f(σ) (reused by DPS's SPSA probe, and later
#     by a Diff-PIR data-consistency step via prox_obj) ---
NMODES_RECON = 25
GN           = true
BLOCK        = true
LAMBDA_GN    = 1e-4
RHO_OBJ      = 1.0e-3

# --- diffusion prior ---
MODEL_KIND = :unet   # :cnn or :unet — which pretrained score network to load
DIM        = 64      # image side length the network was trained on

const βmin = 0.1f0
const βmax = 20.0f0
const T    = 1.0f0

SEED = 1

# -----------------------------------------------------------------------------
# Forward EIT model (mirrors 001SolveTest.ipynb cells 8f99009a..d8719eb6)
# -----------------------------------------------------------------------------

img = load(IMG_PATH)
img_f = img .|> Float64
img_f .-= 0.5
img_f .*= 2.0
img_f .= exp.(img_f)

itp = interpolate_array_2D(Float64.(img_f))

grid = generate_grid(Quadrilateral, (GRID_N, GRID_N))
∂Ω = union(getfacetset.((grid,), ["left", "top", "right", "bottom"])...)
fe = FerriteFESpace{RefQuadrilateral}(grid, 2, 0, 3, ∂Ω)

cond_vec = project_function_to_fem(fe, itp; space=:σ)
cond_vec .= min.(max.(cond_vec, 1e-6), 2.8)

eval_points = reshape(equidistant_grid(DIM), :)
ph = PointEvalHandler(grid, eval_points)

mean_zero_boundary(G_boundary) = G_boundary .- Statistics.mean(G_boundary, dims=1)

G_full = real_fourier_basis(8)
rhs_dict = Dict()
Threads.@threads for i in 2:NUM_MODES+1
    M = make_boundary(G_full[:, i], 64)
    itp_i = interpolate_array_2D(M)
    rhs_dict[i-1] = assemble_rhs_func(fe, itp_i)
end
G = reduce(hcat, [rhs_dict[i] for i in 1:NUM_MODES])
G = fe.up(mean_zero_boundary(fe.down(G)))

fbm_true = FerriteBlockMode(cond_vec, G, fe; block=true)

F = copy(fbm_true.F)
F = mean_zero_boundary(F)
Λ = F * pinv(fe.down(G))
Λ = 0.5 .* (Λ' + Λ)

Ε = randn(size(Λ))
Ε = 0.5 .* (Ε + Ε')
Λn = Λ + SIGMA_NOISE * (norm(Λ) / norm(Ε)) * Ε
Λn = 0.5 .* (Λn + Λn')

F_svd, G_svd, Σ_svd = svd_on_modes(Λn, NUM_MODES)
F_svd = mean_zero_boundary(F_svd)
G_svd = fe.up(mean_zero_boundary(G_svd))

fbm = FerriteBlockMode(F_svd, G_svd, fe)

f, ∂f = create_block_f∂f(fbm, fe; nmodes=NMODES_RECON, block=BLOCK, gn=GN, λ=LAMBDA_GN)
prox_obj = create_prox_linesearch(f, ∂f, RHO_OBJ)

cond_img = reshape(evaluate_at_points(ph, fe.dh_σ, cond_vec), (DIM, DIM))

# -----------------------------------------------------------------------------
# image <-> x_pm1 <-> σ (FEM conductivity) coupling
#
# Three representations of the same DIMxDIM picture show up across these
# reconstruction scripts, and it matters which one a function takes/returns:
#   - a raw image  in [0,1]           (what you'd load from disk / display)
#   - "x_pm1"      in roughly [-1,1]  (the diffusion model's native range)
#   - conductivity in [1e-6, 2.8]     (physical σ, what `f`/`∂f` expect,
#                                      via `exp`/`log` of x_pm1 — see
#                                      cb00f750 in 001SolveTest.ipynb)
# `normalize_image`/`denormalize_image` convert the first pair;
# `image_to_conductivity`/`conductivity_to_x_pm1` convert the second.
# `sigma_from_image`/`conductivity_from_sigma` are the FEM-space versions of
# those, and `project_function_to_fem`/`evaluate_at_points` (via `ph`) are
# the *only* correct ways to move between a 64x64 pixel array and `fe`'s
# conductivity dofs — the dof ordering `fe.dh_σ` uses is Ferrite-internal
# (see how 001SolveTest.ipynb always goes through `ph`/`PointEvalHandler`
# rather than reshaping `cond_vec` directly), so don't be tempted to
# hand-roll a reshape-based resampler.
# -----------------------------------------------------------------------------

normalize_image(img) = 2 .* (Float32.(img) .- 0.5)
denormalize_image(img) = (0.5f0 .* img) .+ 0.5f0

"""
    image_to_conductivity(x_pm1)

`x_pm1` is a (DIM,DIM) array in roughly [-1,1] (the diffusion model's native
range). Maps it to a conductivity field the same way cb00f750 does (exp, then
clamp to the same [1e-6, 2.8] range as `cond_vec`).
"""
function image_to_conductivity(x_pm1::AbstractArray)
    cond = exp.(Float64.(x_pm1))
    return clamp.(cond, 1e-6, 2.8)
end

"""
    conductivity_to_x_pm1(cond)

Inverse of `image_to_conductivity`: `log` of a (clamped) conductivity field,
back into the diffusion model's native x_pm1 range.
"""
function conductivity_to_x_pm1(cond::AbstractArray)
    return Float32.(log.(clamp.(cond, 1e-6, 2.8)))
end

"""
    sigma_from_image(x_pm1)

Projects a (DIM,DIM) diffusion-space image onto the FEM conductivity space
(`fe.dh_σ`), i.e. the σ this notebook's physics loss `f`/`∂f` expect.
"""
function sigma_from_image(x_pm1::AbstractArray)
    cond = image_to_conductivity(x_pm1)
    itp_x = interpolate_array_2D(cond)
    σ = project_function_to_fem(fe, itp_x; space=:σ)
    return clamp.(σ, 1e-6, 2.8)
end

"""
    conductivity_from_sigma(σ)

Inverse direction: evaluates an FEM conductivity vector back onto the
DIMxDIM pixel grid (same as 001SolveTest.ipynb's `cond_img`/`σ_img` cells).
Returns raw positive conductivity — *not* x_pm1 or display range, see
`x_pm1_from_sigma`/`display_image_from_sigma` below for those.
"""
function conductivity_from_sigma(σ::AbstractVector)
    return reshape(evaluate_at_points(ph, fe.dh_σ, σ), (DIM, DIM))
end

"""
    x_pm1_from_sigma(σ)

`conductivity_from_sigma` composed with `conductivity_to_x_pm1` — maps an FEM
conductivity vector back into the diffusion model's native x_pm1 space, e.g.
to feed a Tweedie-corrected σ̂₀ back into `sde` (Diff-PIR step 5).
"""
x_pm1_from_sigma(σ::AbstractVector) = conductivity_to_x_pm1(conductivity_from_sigma(σ))

"""
    display_image_from_sigma(σ)

`x_pm1_from_sigma` composed with `denormalize_image` — a [0,1]-ish array
ready for `Gray.()`/plotting. Replaces the `log.(...) .* 0.5 .+ 0.5` formula
that used to be duplicated across dps_reconstruct.jl/diffpir_reconstruct.jl/
reddiff_reconstruct.jl.
"""
display_image_from_sigma(σ::AbstractVector) = denormalize_image(x_pm1_from_sigma(σ))

"""
    display_image_from_conductivity(cond)

Same as `display_image_from_sigma` but starting from a raw conductivity
array instead of an FEM σ vector — e.g. for `cond_img`, the ground-truth
conductivity map used as `true_img` in the reconstruction scripts.
"""
display_image_from_conductivity(cond::AbstractArray) = denormalize_image(conductivity_to_x_pm1(cond))

"""
    data_fidelity(x_pm1)

Scalar EIT data-fidelity loss of a diffusion-space image: `f(σ(x_pm1))`.
This is the function DPS/Diff-PIR/RED-Diff all want a gradient of w.r.t.
`x_pm1` — see `spsa_grad` below for how that gradient gets estimated.
"""
data_fidelity(x_pm1::AbstractArray) = f(sigma_from_image(x_pm1))

"""
    spsa_grad(h, x; c=0.05f0, n_probes=1)

Simultaneous Perturbation Stochastic Approximation gradient estimate of a
scalar function `h` at `x`: 2 evaluations of `h` per probe, regardless of
`length(x)`.

Why SPSA instead of exact backprop here: `sigma_from_image` runs a bilinear
resample (Interpolations.jl) through an FE L² projection (mutating Ferrite
assembly + a sparse solve) — plumbing that isn't set up to be pushed through
Zygote/Enzyme, and hand-deriving its exact adjoint is a separate project of
its own (`project_function_to_fem` in
src/Galerkin/Ferrite/Assemblers/RHSAssemblers.jl is linear in the pixel
array, so an exact adjoint *is* possible, just not implemented). SPSA gives
an unbiased gradient estimate using only forward evaluations of `f`, which
is already fast (adjoint-cached physics solve) — trade exactness for not
having to write and validate new FEM assembly code.

Increase `n_probes` to trade compute for a lower-variance estimate.
"""
function spsa_grad(h, x::AbstractArray{T}; c::Real=0.05, n_probes::Int=1) where {T}
    # `c::Real` (not `c::T`) deliberately: a keyword type tied to a function's
    # `where T` is checked with `isa`, not `convert`, so a literal like 0.05f0
    # throws unless T happens to match exactly. Converting explicitly avoids that.
    cT = T(c)
    g = zero(x)
    for _ in 1:n_probes
        Δ = rand(T[-1, 1], size(x))
        h_plus = h(x .+ cT .* Δ)
        h_minus = h(x .- cT .* Δ)
        g .+= ((h_plus - h_minus) / (2cT)) .* Δ   # 1/Δ == Δ since Δ ∈ {-1,1}
    end
    return g ./ n_probes
end

# -----------------------------------------------------------------------------
# Diffusion prior: pretrained VP-SDE score network (CNN or U-Net)
# -----------------------------------------------------------------------------

const xdev = reactant_device(; force=true)
const cdev = cpu_device()
dev = xdev

if MODEL_KIND === :cnn
    include(joinpath(@__DIR__, "CNN", "model.jl"))
    emb_dim = 64
    model = local_score_net_64(; embedding_dims=emb_dim)
    ckpt_dir = joinpath(@__DIR__, "CNN")
elseif MODEL_KIND === :unet
    include(joinpath(@__DIR__, "Unet", "model.jl"))
    emb_dim = 32
    model = unet_tinyimagenet64(; embedding_dims=emb_dim)
    ckpt_dir = joinpath(@__DIR__, "Unet")
else
    throw(ArgumentError("MODEL_KIND must be :cnn or :unet, got $MODEL_KIND"))
end

ps_path = joinpath(ckpt_dir, "ps_latestvn.jld2")
st_path = joinpath(ckpt_dir, "st_latestvn.jld2")
if isfile(ps_path) && isfile(st_path)
    @load ps_path ps_cpu
    @load st_path st_cpu
    st_cpu = Lux.testmode(st_cpu)
    ps = ps_cpu |> dev
    st = st_cpu |> dev
    println("Loaded checkpoint from $ps_path / $st_path.")
else
    @warn "No checkpoint found at $ckpt_dir — sampling from a randomly initialized model."
    ps_cpu, st_cpu = Lux.setup(Xoshiro(SEED), model)
    st_cpu = Lux.testmode(st_cpu)
    ps = ps_cpu |> dev
    st = st_cpu |> dev
end
# ps_cpu/st_cpu (plain CPU arrays) back `sde` below; ps/st (on `dev`) back R_diff.
# Keeping sde on CPU sidesteps a Reactant scalar-indexing crash on batch-size-1 calls.

β(t) = βmin + (βmax - βmin) * t / T
ᾱ(t) = exp(-βmin * t - (βmax - βmin) / (2 * T) * t^2)

"""
    sde(x, t)

Evaluate the trained noise predictor ε̂(x_t, t) for a single DIMxDIM
grayscale image `x` at scalar diffusion time `t ∈ [0, T]`. Dispatches on
`MODEL_KIND` since the CNN and U-Net take their time argument differently
(a plain Vector vs. a (1,1,1,1) array) — see ../CNN/sample.jl / ../Unet/sample.jl.
"""
function sde(x::AbstractArray, t::Real)
    x4 = reshape(Float32.(x), DIM, DIM, 1, 1)
    if MODEL_KIND === :cnn
        tv = Float32[t]
        ε̂, _ = model((x4, tv), ps_cpu, st_cpu)
    else
        nv = fill(Float32(t), 1, 1, 1, 1)
        ε̂, _ = model((x4, nv), ps_cpu, st_cpu)
    end
    return reshape(Array(ε̂), DIM, DIM)
end

"""
    R_diff(x, T, n; t_min=0.02f0, w=t -> 1.0f0)

RED-Diff regularizer (stop-gradient through the network) — identical to the
one in ../CNN/sample.jl / ../Unet/sample.jl, kept here so Diff-PIR/RED-Diff
scripts can reuse it via this same initialize.jl. See those files for the
full derivation; unchanged here.
"""
function R_diff(x::AbstractArray, T::Real, n::Int; t_min::Real=0.02f0, w=t -> 1.0f0)
    x2 = Float32.(reshape(x, DIM, DIM))
    ts = t_min .+ (T - t_min) .* rand(Float32, n)
    ε = randn(Float32, DIM, DIM, 1, n)

    αbar = reshape(Float32.(ᾱ.(ts)), 1, 1, 1, n)
    x_batch = reshape(x2, DIM, DIM, 1, 1) .* sqrt.(αbar) .+ sqrt.(1 .- αbar) .* ε

    if MODEL_KIND === :cnn
        ε̂, _ = model((x_batch |> dev, ts |> dev), ps, st)
    else
        nv = reshape(Float32.(ts), 1, 1, 1, n)
        ε̂, _ = model((x_batch |> dev, nv |> dev), ps, st)
    end
    residual = Array(ε̂ |> cdev) .- ε
    err = sum(abs2, residual) / length(residual)

    weights = reshape(Float32.(w.(ts)), 1, 1, 1, n) .* sqrt.(αbar)
    grad = dropdims(sum(weights .* residual; dims=4); dims=(3, 4)) ./ n

    return err, grad
end
