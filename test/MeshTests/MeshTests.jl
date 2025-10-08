using ModularEIT
using Ferrite
using FerriteGmsh

@testset "MeshTests" begin
    output_string = create_circle_mesh(0.05, 1)
    println(output_string)

    @test length(output_string)> 20


end
