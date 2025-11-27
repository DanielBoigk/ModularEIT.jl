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

export collect_Jr!

function collect_Jr!(problem::FerriteProblem, n::Int)
    problem.state.opt.J = zeros(n, problem.fe.n)
    problem.state.opt.r = zeros(n)

    for i in 1:n
        problem.state.opt.J[i, :] = problem.modes[i].δσ
        problem.state.opt.r[i] = problem.modes[i].error_n
    end
end

export update_all!

function update_all!(problem::FerriteProblem, n::Int)
    collect_Jr!(problem, n)
    gauss_newton_lm_cg!(problem.state.opt, 5000)
    problem.state.δ = problem.state.opt.δ
    update_sigma!(problem.state)
    update_L!(problem.state, problem.fe)
end
