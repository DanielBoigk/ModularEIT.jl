
function solve_modes()



function solve_modes!(J::AbstractMatrix,r::AbstractVector, modes::Dict{Int,FerriteEITMode},fe::FerriteFESpace, sol::FerriteSolutionState, solver_func!; maxiter = 500)
    @threads for i in eachindex(modes)
        solver_func!(modes[i],sol,fe,maxiter)
    end
end


#=
function solve_modes!(J::AbstractMatrix,r::AbstractVector, modes::Dict{Int,FerriteEITMode},fe::FerriteFESpace, sol::FerriteSolutionState,num_modes::Int)
=#
