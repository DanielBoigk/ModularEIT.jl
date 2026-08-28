using Images, FileIO, Dates

include("sample.jl")   # brings in model, ps_cpu, st_cpu, sde, β, ᾱ, dim, T

# =============================================================================
# test_unconditional.jl — pure unconditional Euler-Maruyama sampling from the
# CNN (LocalScoreNet) checkpoint, with *no* EIT physics guidance at all.
#
# Purpose: isolate whether the "coarse noise" seen in dps_reconstruct.jl
# comes from a bad interaction between the physics gradient and the
# diffusion prior, or whether the CNN prior itself can't produce a coherent
# image even unconditionally (e.g. undertrained checkpoint, or the
# architecture's deliberately bounded receptive field just can't do this).
# =============================================================================

N_STEPS = 1000
T_MIN   = 1.0f-3

OUT_DIR = joinpath(@__DIR__, "unconditional_samples")
mkpath(OUT_DIR)

ts = Float32.(range(T, T_MIN; length=N_STEPS))

x = randn(Float32, dim, dim)   # x_T ~ N(0, I)

for i in 1:N_STEPS
    global x
    t = ts[i]
    t_next = i < N_STEPS ? ts[i+1] : 0.0f0
    Δt = t - t_next

    ε̂ = sde(x, t)
    αbar_t = ᾱ(t)
    β_t = β(t)
    score = -ε̂ ./ sqrt(1 - αbar_t)
    drift = -0.5f0 .* β_t .* x .- β_t .* score
    noise = i < N_STEPS ? sqrt(β_t * Δt) .* randn(Float32, dim, dim) : zero(x)
    x = x .- Δt .* drift .+ noise

    if i % 100 == 0 || i == N_STEPS
        println("step $i/$N_STEPS  t=$(round(t, digits=3))")
    end
end

img = Gray.(map(clamp01nan, denormalize_image(x)))
timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
out_path = joinpath(OUT_DIR, "unconditional_sample_$timestamp.png")
save(out_path, img)
println("Saved unconditional sample to $out_path")
