mutable struct FerriteFEMatrices # Maybe that should not be specific to Ferrite
# Union{Nothing, AbstractMatrix}
# massmatrix
# stiffness matrix
# potentially GPU versions thereof
end

# I'm unsure whether this should be mutable. Maybe one can just build a new FESpace any time the mesh get's refined/coarsened.
struct  FerriteFESpaces <: AbstractHilbertSpaces
    # grid
    # single dofhandler consolidating σ and u
    # Please keep this agnostic w.r.t.
    # cellvalues

end

