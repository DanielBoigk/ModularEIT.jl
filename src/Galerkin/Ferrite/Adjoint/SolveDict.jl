
export solve_modes!



function solve_modes!(state::FerriteSolverState, solver_func!, maxiter = 500)
    fe = state.fe

    @threads for (i,mode) in state.modes
        solver_func!(mode, , fe, maxiter)
    end



end



#=
function solve_modes!(J::AbstractMatrix,r::AbstractVector, modes::Dict{Int,FerriteEITMode},fe::FerriteFESpace, sol::FerriteSolutionState,num_modes::Int)
=#
