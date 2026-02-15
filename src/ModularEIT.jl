module ModularEIT

include("Galerkin/Galerkin.jl")
export printsomething

include("Glue/mesh_creator.jl")
include("Images/Interpolate.jl")
function printsomething()
    println("Hallo Welt!")
    return "some output"
end
greet() = print("Hello World!")

end # module EIT4Julia
include("ImageHandler/ImageHandler.jl")
