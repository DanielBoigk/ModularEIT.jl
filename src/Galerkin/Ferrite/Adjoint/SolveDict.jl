export solve_modes!

function solve_modes!(problem::FerriteProblem, n::Int, solver_func!, maxiter=500)
    fe = problem.fe
    modes = problem.modes
    sol = problem.state
    Threads.@threads for i in 1:n
        solver_func!(modes[i], sol, fe, maxiter)
    end
end

function solve_modes!(problem::FerriteProblem, n::Int)
    fe = problem.fe
    modes = problem.modes
    sol = problem.state
    Threads.@threads for i in 1:n
        state_adjoint_step_neumann_cg!(modes[i], sol, fe, 500)
    end
end


#=
function solve_modes!(state::FerriteSolverState, solver_func!, maxiter = 500)
    fe = state.fe

    @threads for (i,mode) in state.modes
        solver_func!(mode, , fe, maxiter)
    end



end
=#


#=
function solve_modes!(J::AbstractMatrix,r::AbstractVector, modes::Dict{Int,FerriteEITMode},fe::FerriteFESpace, sol::FerriteSolutionState,num_modes::Int)
=#
