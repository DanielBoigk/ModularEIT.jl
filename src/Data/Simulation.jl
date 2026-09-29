# Putting phantoms on a discretization and simulating (noisy) measurements.

"""
    conductivity(disc, phantom; method = :l2, quadrature_order = 4)
    conductivity(disc, σ::AbstractVector)

Coefficients of a conductivity given as a function `x ↦ σ(x)` (a phantom, e.g.
[`InclusionPhantom`](@ref), [`PixelFunction`](@ref), or any callable) in the σ space of `disc`:
the L² projection (`method = :l2`, cell averages for piecewise constants, sampled with a rule of
`quadrature_order`) or nodal interpolation (`:interpolate`). Coefficient vectors are passed
through.
"""
function conductivity(disc::AbstractDiscretization, σ::AbstractVector; kwargs...)
    length(σ) == ndofs_σ(disc) || throw(DimensionMismatch("σ must have $(ndofs_σ(disc)) coefficients"))
    return σ
end
function conductivity(disc::AbstractDiscretization, phantom; method::Symbol = :l2, quadrature_order::Integer = 4)
    method === :l2 && return l2_project(disc, phantom; quadrature_order)
    method === :interpolate && return interpolate_function(disc, phantom)
    throw(ArgumentError("method must be :l2 or :interpolate, got :$method"))
end

"""
    simulate_data(disc, fm, conductivity, inputs; mode = :neumann, noise = nothing,
                  rng = Random.default_rng(), solver = DirectSolver(), quadrature_order = 4)

Simulated measurements for the forward model `fm` on `disc`: `conductivity` is a phantom
(projected with [`conductivity`](@ref)) or a coefficient vector; `inputs` are current patterns
(`mode = :neumann`, data = voltages) or voltage patterns (`mode = :dirichlet`, data = currents).
`noise`: `nothing`, an additive [`AbstractNoiseModel`](@ref) or [`SourceMeterNoise`](@ref).

Returns a named tuple `(data, clean, σ, inputs, applied)`: noisy and exact data, the conductivity
coefficients, the nominal inputs (to give to the reconstruction) and the inputs actually applied
(different under source noise).

To avoid the inverse crime, simulate on a finer (or different) mesh than the reconstruction
mesh: electrode patterns and electrode voltages do not depend on the mesh.
"""
function simulate_data(disc::AbstractDiscretization, fm::ForwardModel, cond, inputs::AbstractVecOrMat;
                       mode::Symbol = :neumann, noise::Union{Nothing, AbstractNoiseModel} = nothing,
                       rng::AbstractRNG = Random.default_rng(), solver::AbstractLinearSolver = DirectSolver(),
                       quadrature_order::Integer = 4)
    mode in (:neumann, :dirichlet) || throw(ArgumentError("mode must be :neumann or :dirichlet, got :$mode"))
    σ = conductivity(disc, cond; quadrature_order)
    forward(u) = mode === :neumann ? forward_neumann(fm, σ, u; solver)[1] : forward_dirichlet(fm, σ, u; solver)[1]
    clean = forward(inputs)
    if noise isa SourceMeterNoise
        applied = iszero(noise.source.std) ? copy(inputs) : add_noise(inputs, noise.source; rng)
        if mode === :neumann && !iszero(noise.source.std)     # zero net current, as the nominal patterns
            w = vec(sum(fm.P; dims = 1))
            applied = applied .- (w' * applied) ./ sum(w)
        end
        data = add_noise!(forward(applied), noise.meter; rng)
    else
        applied = copy(inputs)
        data = noise === nothing ? copy(clean) : add_noise(clean, noise; rng)
    end
    return (; data, clean, σ, inputs, applied)
end
