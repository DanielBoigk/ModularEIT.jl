# Electrode models: how current enters and voltage is measured on the boundary. The models only
# hold the electrodes (in the back end's representation, e.g. vectors of Ferrite `FacetIndex`)
# and parameters; a back end turns a model on a discretization into a ForwardModel.
#
#   continuum:  every boundary dof is an electrode. Currents are boundary current densities
#               (coefficients at the boundary dofs), P g = ∫_Γ g φᵢ; voltages are the boundary
#               values of u.
#   point:      current enters at single boundary nodes, voltages are read at single nodes.
#   gap:        current I_ℓ is spread uniformly over electrode e_ℓ, P = ∫_{e_ℓ} φᵢ / |e_ℓ|;
#               the voltage is the mean potential on the electrode (Q = Pᵀ for the same
#               electrodes). Injection and measurement electrodes may differ.
#   complete (CEM): the electrode voltages U_ℓ are extra unknowns, contact impedances z_ℓ;
#               the voltage is measured on the electrodes, including current-carrying ones.

"""
    ContinuumModel()

Continuum model: every boundary dof acts as an electrode. Current patterns are boundary current
densities (one value per boundary dof, `disc.boundary_dofs` order), voltages are the boundary
values of the potential.
"""
struct ContinuumModel <: AbstractElectrodeModel end

"""
    PointElectrodeModel(points; measure = points)

Point electrodes: current `I_ℓ` enters at the boundary dof closest to `points[ℓ]`, voltages are
read at the boundary dofs closest to `measure`. Injection and measurement points may differ.
"""
struct PointElectrodeModel{V} <: AbstractElectrodeModel
    inject::Vector{V}
    measure::Vector{V}
end
PointElectrodeModel(points; measure = points) = PointElectrodeModel(collect(points), collect(measure))

"""
    GapModel(electrodes; measure = electrodes)

Gap model: current `I_ℓ` is uniformly distributed over electrode `electrodes[ℓ]` (a vector of
boundary facets in the back end's representation, e.g. `FacetIndex` for Ferrite), zero in
the gaps. The voltage of a measurement electrode is the mean
potential on it. Injection and measurement electrodes may differ (`measure`).
"""
struct GapModel{E} <: AbstractElectrodeModel
    inject::Vector{E}
    measure::Vector{E}
end
function GapModel(electrodes; measure = electrodes)
    inject, meas = collect.(electrodes), collect.(measure)
    E = promote_type(eltype(inject), eltype(meas))
    return GapModel{E}(inject, meas)
end

"""
    CompleteElectrodeModel(electrodes, z; measure = 1:length(electrodes))

Complete electrode model with contact impedances `z` (scalar or one per electrode). The electrode
voltages `U` are unknowns of the forward problem (grounded by `Σ U_ℓ = 0`); `measure` selects the
electrodes whose voltages are measured (e.g. only electrodes that carry no current).
"""
struct CompleteElectrodeModel{E} <: AbstractElectrodeModel
    electrodes::Vector{E}
    z::Vector{Float64}
    measure::Vector{Int}
end
function CompleteElectrodeModel(electrodes, z; measure = 1:length(electrodes))
    L = length(electrodes)
    zv = z isa Number ? fill(Float64(z), L) : collect(Float64, z)
    length(zv) == L || throw(DimensionMismatch("need one contact impedance per electrode"))
    all(>(0), zv) || throw(ArgumentError("contact impedances must be positive"))
    all(in(1:L), measure) || throw(ArgumentError("measure must index electrodes 1:$L"))
    return CompleteElectrodeModel(collect.(electrodes), zv, collect(measure))
end
