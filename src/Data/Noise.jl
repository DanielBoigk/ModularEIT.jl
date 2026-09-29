# Noise models for simulated EIT data (wiki: Noise Models for EIT Data, Inverse Crime).
#
# Data are matrices with one column per pattern (vectors are one pattern). Additive models draw
# independent Gaussian noise with a standard deviation per entry:
#     GaussianNoise(s)          s absolute (scalar, or one value per measurement channel)
#     RelativeGaussianNoise(δ)  s = δ ‖fᵢ‖ / √m for column fᵢ, so that E‖ηᵢ‖² = δ² ‖fᵢ‖²
# SourceMeterNoise perturbs the inputs (current source) before the forward solve and the outputs
# (voltmeter) after it, so it needs the forward model: see simulate_data.
#
# Modelling errors (often larger than instrument noise) are simulated by generating the data with
# a different model than the reconstruction: finer mesh (simulate_data), perturbed contact
# impedances (perturb_contact_impedance) and electrode positions (electrode_angles).

using Random

"""
    GaussianNoise(s)

Additive white Gaussian noise with standard deviation `s`: a scalar, or a vector with one
standard deviation per measurement channel (data row), e.g. from an instrument specification.
"""
struct GaussianNoise{S <: Union{Float64, Vector{Float64}}} <: AbstractNoiseModel
    std::S
    function GaussianNoise(s::Union{Real, AbstractVector{<:Real}})
        all(>=(0), s) || throw(ArgumentError("standard deviations must be nonnegative"))
        v = s isa Real ? Float64(s) : Vector{Float64}(s)
        return new{typeof(v)}(v)
    end
end

"""
    RelativeGaussianNoise(δ)

Additive Gaussian noise relative to the signal level: for every pattern (column) `fᵢ` with `m`
entries, `ηᵢ ~ N(0, (δ ‖fᵢ‖ / √m)² I)`, so that `‖ηᵢ‖ ≈ δ ‖fᵢ‖` (e.g. `δ = 0.01` for 1 % noise).
"""
struct RelativeGaussianNoise <: AbstractNoiseModel
    δ::Float64
    function RelativeGaussianNoise(δ::Real)
        δ >= 0 || throw(ArgumentError("the relative noise level must be nonnegative"))
        return new(Float64(δ))
    end
end

"""
    SourceMeterNoise(source, meter)

Errors of the current source and of the voltmeter (or, for voltage-driven data, of the voltage
source and the ammeter), each a [`GaussianNoise`](@ref) standard deviation (scalar or per
channel): the applied inputs are `inputs + source ε` (projected back to zero net current for
current patterns), the measurements `F(applied) + meter ε`. The reconstruction is given the
nominal inputs. Needs the forward model, so it is applied by [`simulate_data`](@ref).
"""
struct SourceMeterNoise{S, M} <: AbstractNoiseModel
    source::GaussianNoise{S}
    meter::GaussianNoise{M}
end
SourceMeterNoise(source, meter) = SourceMeterNoise(GaussianNoise(source), GaussianNoise(meter))

# standard deviation of every entry of `data` (broadcastable against it)
function _noise_std(noise::GaussianNoise{Float64}, data::AbstractVecOrMat)
    return noise.std
end
function _noise_std(noise::GaussianNoise{Vector{Float64}}, data::AbstractVecOrMat)
    length(noise.std) == size(data, 1) ||
        throw(DimensionMismatch("the noise model has $(length(noise.std)) channels, the data $(size(data, 1)) rows"))
    return noise.std
end
function _noise_std(noise::RelativeGaussianNoise, data::AbstractVecOrMat)
    m = size(data, 1)
    return noise.δ .* sqrt.(sum(abs2, data; dims = 1)) ./ sqrt(m)
end
_noise_std(::SourceMeterNoise, data) =
    throw(ArgumentError("SourceMeterNoise perturbs the inputs of the forward problem; use simulate_data"))

"""
    add_noise(data, noise; rng = Random.default_rng())
    add_noise!(data, noise; rng = Random.default_rng())

Noisy copy of `data` (or `data` overwritten) for an additive noise model
([`GaussianNoise`](@ref), [`RelativeGaussianNoise`](@ref)). Pass a seeded `rng` for
reproducible noise.
"""
add_noise(data::AbstractVecOrMat, noise::AbstractNoiseModel; rng::AbstractRNG = Random.default_rng()) =
    add_noise!(Array{Float64}(data), noise; rng)

function add_noise!(data::AbstractVecOrMat, noise::AbstractNoiseModel; rng::AbstractRNG = Random.default_rng())
    s = _noise_std(noise, data)
    data .+= s .* randn(rng, size(data))
    return data
end

"""
    expected_squared_error(noise, data)

Expected squared norm `E‖η‖²` (summed over all entries) of the noise that `noise` adds to
`data`.
"""
expected_squared_error(noise::AbstractNoiseModel, data::AbstractVecOrMat) =
    sum(abs2.(_noise_std(noise, data)) .* ones(size(data)))

"""
    discrepancy_target(obj::AdjointStateObjective, noise; τ = 1.1)

Target value of the objective for the discrepancy principle: `τ² E[J(σ_true)]`, the expected
misfit of exact data corrupted by `noise`, taken in the objective's metric (the whitening of its
misfit and, for voltages, the removal of the weighted mean). Pass it as `ftarget` to
[`minimize`](@ref) to stop when the data are explained to the noise level (`τ` slightly above 1).
Relative noise levels are evaluated on the objective's (noisy) data.
"""
function discrepancy_target(obj::AdjointStateObjective, noise::AbstractNoiseModel; τ::Real = 1.1)
    data = obj.data
    n_obs = size(data, 1)
    U = Matrix(_whitening_adjoint_matrix(obj.misfit, n_obs)')
    B = if obj.mode === :neumann
        w = obj.fm.measure_weights
        U * (I - ones(n_obs) * w' ./ sum(w))                   # U Π
    else
        U
    end
    colnorm2 = vec(sum(abs2, B; dims = 1))                     # ‖B eⱼ‖²
    S = abs2.(_noise_std(noise, data)) .* ones(size(data))      # variances, n_obs × s
    return τ^2 * dot(colnorm2, vec(sum(S; dims = 2))) / 2
end

"""
    perturb_boundary_operator(R, s; rng = Random.default_rng())

Operator-level noise for a discrete boundary operator (e.g. an estimated Neumann-to-Dirichlet
matrix): `R + s (Ê + Êᵀ)/2` with i.i.d. standard normal `Ê`. Symmetry is preserved, positive
semidefiniteness is not (for large `s`).
"""
function perturb_boundary_operator(R::AbstractMatrix, s::Real; rng::AbstractRNG = Random.default_rng())
    n = LinearAlgebra.checksquare(R)
    E = randn(rng, n, n)
    return Matrix{Float64}(R) .+ s .* (E .+ E') ./ 2
end

"""
    perturb_contact_impedance(model::CompleteElectrodeModel, s; rng = Random.default_rng())

Complete electrode model with log-normally perturbed contact impedances `zₗ exp(s εₗ)`, a
modelling error for data simulation (the reconstruction keeps the nominal model).
"""
function perturb_contact_impedance(model::CompleteElectrodeModel, s::Real; rng::AbstractRNG = Random.default_rng())
    z = model.z .* exp.(s .* randn(rng, length(model.z)))
    return CompleteElectrodeModel(model.electrodes, z; measure = model.measure)
end

"""
    electrode_angles(L; offset = 0, jitter = 0, rng = Random.default_rng())

Centre angles `offset + 2π(ℓ-1)/L + jitter εₗ` of `L` electrodes, with Gaussian position errors
of standard deviation `jitter` (radians); pass them as `angles` to [`angular_electrodes`](@ref).
"""
function electrode_angles(L::Integer; offset::Real = 0.0, jitter::Real = 0.0, rng::AbstractRNG = Random.default_rng())
    θ = offset .+ 2π .* (0:(L - 1)) ./ L
    return jitter == 0 ? collect(θ) : θ .+ jitter .* randn(rng, L)
end

# mean standard deviation of the whitened residual (for posterior_std)
_residual_noise_std(obj::AdjointStateObjective, noise::AbstractNoiseModel) =
    sqrt(2 * discrepancy_target(obj, noise; τ = 1) / n_residual(obj))
