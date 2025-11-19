
include("FerriteStructs.jl")

include("FESpace/FESpace.jl")
include("Assemblers/MatrixAssemblers.jl")
include("Assemblers/RHSAssemblers.jl")
include("Assemblers/TensorAssembler.jl")

include("Projectors/Boundary.jl")
#include("Projectors/Volume.jl")

include("SolverState/SolverState.jl")
include("Adjoint/Adjoint.jl")
