module ModularEIT

include("Galerkin/Galerkin.jl")
export printsomething

function printsomething()
    println("Hallo Welt!")
    return "some output"
end
greet() = print("Hello World!")

end # module EIT4Julia
